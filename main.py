import argparse
import ast
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

from z3 import (
    And,
    Array,
    ArraySort,
    BitVecSort,
    BitVecVal,
    BoolSort,
    Const,
    DeclareSort,
    Distinct,
    EnumSort,
    Exists,
    ExprRef,
    ForAll,
    FuncDeclRef,
    Function,
    If,
    Implies,
    IntSort,
    Not,
    Optimize,
    Or,
    Product,
    RealSort,
    Solver,
    SortRef,
    Sum,
    sat,
    unsat,
)

# Setup logging - only configure if running as main script
logger = logging.getLogger(__name__)


# Abstract Solver Interface
class AbstractSolver(ABC):
    """Abstract base class for solver implementations."""

    @abstractmethod
    def add(self, constraint: Any) -> None:
        """Add a constraint to the solver."""
        raise NotImplementedError

    @abstractmethod
    def check(self, condition: Any = None) -> Any:
        """Check satisfiability of constraints."""
        raise NotImplementedError

    @abstractmethod
    def model(self) -> Any:
        """Get the model if SAT."""
        raise NotImplementedError

    @abstractmethod
    def set(self, param: str, value: Any) -> None:
        """Set solver parameter."""
        raise NotImplementedError


# Z3 Solver Implementation
class Z3Solver(AbstractSolver):
    """Z3 solver implementation."""

    def __init__(self) -> None:
        self.solver = Solver()

    def add(self, constraint: Any) -> None:
        """Add a constraint to the Z3 solver."""
        self.solver.add(constraint)

    def check(self, condition: Any = None) -> Any:
        """Check satisfiability with optional condition."""
        if condition is not None:
            return self.solver.check(condition)
        return self.solver.check()

    def model(self) -> Any:
        """Return the satisfying model."""
        return self.solver.model()

    def set(self, param: str, value: Any) -> None:
        """Set Z3 solver parameter."""
        self.solver.set(param, value)


class Z3JSONInterpreter:
    """Interpreter for Z3 DSL defined in JSON format."""

    # Default timeout values in milliseconds
    DEFAULT_VERIFY_TIMEOUT = 10000
    DEFAULT_OPTIMIZE_TIMEOUT = 100000
    MAX_BITVEC_SIZE = 65536  # Maximum reasonable bitvector size

    # Safe expression evaluation globals
    Z3_OPERATORS = {
        "And": And,
        "Or": Or,
        "Not": Not,
        "Implies": Implies,
        "If": If,
        "Distinct": Distinct,
        "Sum": Sum,
        "Product": Product,
        "ForAll": ForAll,
        "Exists": Exists,
        "Function": Function,
        "Array": Array,
        "BitVecVal": BitVecVal,
    }

    def __init__(
        self,
        json_file: str,
        solver: AbstractSolver | None = None,
        verify_timeout: int = DEFAULT_VERIFY_TIMEOUT,
        optimize_timeout: int = DEFAULT_OPTIMIZE_TIMEOUT,
    ):
        """Initialize the Z3 JSON interpreter.

        Args:
            json_file: Path to JSON configuration file
            solver: Optional solver instance (defaults to Z3Solver)
            verify_timeout: Timeout for verification in milliseconds
            optimize_timeout: Timeout for optimization in milliseconds
        """
        self.json_file = json_file
        self.verify_timeout = verify_timeout
        self.optimize_timeout = optimize_timeout
        self.config = self.load_and_validate_json(json_file)
        self.solver = solver if solver else Z3Solver()
        self.optimizer = Optimize()
        self.sorts: dict[str, SortRef] = {}
        self.functions: dict[str, FuncDeclRef] = {}
        self.constants: dict[str, Any] = {}
        self.variables: dict[str, Any] = {}
        self.verifications: dict[str, ExprRef] = {}
        self._context_cache: dict[str, Any] | None = None
        self._symbols_loaded = False  # Track if all symbols have been loaded

    def load_and_validate_json(self, json_file: str) -> dict[str, Any]:
        """Load and validate JSON configuration file.

        Args:
            json_file: Path to JSON file

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            ValueError: If required sections are invalid
        """
        try:
            with open(json_file) as file:
                config = json.load(file)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {json_file}: {e}")
            raise

        # Initialize missing sections with appropriate defaults
        default_sections: dict[str, Any] = {
            "sorts": [],
            "functions": [],
            "constants": {},
            "knowledge_base": [],
            "rules": [],
            "verifications": [],
            "actions": [],
            "variables": [],
        }

        for section, default in default_sections.items():
            if section not in config:
                config[section] = default
                logger.debug(f"Section '{section}' not found, using default: {default}")

        # Validate structure
        if not isinstance(config.get("constants"), dict):
            config["constants"] = {}
            logger.warning("'constants' section should be a dictionary, resetting to empty dict")

        return config

    def _topological_sort_sorts(self, sort_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Topologically sort sort definitions to handle dependencies.

        Args:
            sort_defs: List of sort definitions

        Returns:
            Sorted list where dependencies come before dependents

        Raises:
            ValueError: If circular dependency detected
        """
        # Build dependency graph
        dependencies = {}
        for sort_def in sort_defs:
            name = sort_def["name"]
            sort_type = sort_def["type"]
            deps = []

            # Extract dependencies based on sort type
            if sort_type.startswith("ArraySort("):
                domain_range = sort_type[len("ArraySort(") : -1]
                parts = [s.strip() for s in domain_range.split(",")]
                deps.extend(parts)

            dependencies[name] = deps

        # Perform topological sort using Kahn's algorithm
        in_degree = {name: 0 for name in dependencies}
        for deps in dependencies.values():
            for dep in deps:
                if dep in in_degree:  # Only count dependencies that are user-defined
                    in_degree[dep] += 1

        # Start with nodes that have no dependencies (or only built-in dependencies)
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_names = []

        while queue:
            current = queue.pop(0)
            sorted_names.append(current)

            # Reduce in-degree for dependents
            for name, deps in dependencies.items():
                if current in deps and name not in sorted_names:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        # Check for cycles
        if len(sorted_names) != len(dependencies):
            remaining = set(dependencies.keys()) - set(sorted_names)
            raise ValueError(f"Circular dependency detected in sorts: {remaining}")

        # Reorder sort_defs according to sorted_names
        name_to_def = {s["name"]: s for s in sort_defs}
        return [name_to_def[name] for name in sorted_names]

    def create_sorts(self) -> None:
        """Create Z3 sorts from configuration.

        Raises:
            ValueError: If sort definition is invalid
        """
        # Add built-in sorts
        built_in_sorts = {"BoolSort": BoolSort(), "IntSort": IntSort(), "RealSort": RealSort()}
        self.sorts.update(built_in_sorts)

        # Topologically sort sorts to handle dependencies
        sorted_sort_defs = self._topological_sort_sorts(self.config["sorts"])

        # Create user-defined sorts in dependency order
        for sort_def in sorted_sort_defs:
            try:
                name = sort_def["name"]
                sort_type = sort_def["type"]

                if sort_type == "EnumSort":
                    values = sort_def["values"]
                    enum_sort, enum_consts = EnumSort(name, values)
                    self.sorts[name] = enum_sort
                    # Add enum constants to context
                    for val_name, const in zip(values, enum_consts, strict=False):
                        self.constants[val_name] = const
                elif sort_type.startswith("BitVecSort("):
                    size_str = sort_type[len("BitVecSort(") : -1].strip()
                    try:
                        size = int(size_str)
                        if size <= 0:
                            raise ValueError(f"BitVecSort size must be positive, got {size}")
                        if size > self.MAX_BITVEC_SIZE:
                            raise ValueError(
                                f"BitVecSort size {size} exceeds maximum {self.MAX_BITVEC_SIZE}"
                            )
                        self.sorts[name] = BitVecSort(size)
                    except ValueError as e:
                        raise ValueError(f"Invalid BitVecSort size '{size_str}': {e}") from e
                elif sort_type.startswith("ArraySort("):
                    domain_range = sort_type[len("ArraySort(") : -1]
                    domain_sort_name, range_sort_name = [s.strip() for s in domain_range.split(",")]
                    domain_sort = self.sorts.get(domain_sort_name)
                    range_sort = self.sorts.get(range_sort_name)
                    if not domain_sort or not range_sort:
                        raise ValueError(
                            f"ArraySort references undefined sorts: {domain_sort_name}, {range_sort_name}"
                        )
                    self.sorts[name] = ArraySort(domain_sort, range_sort)
                elif sort_type == "IntSort":
                    self.sorts[name] = IntSort()
                elif sort_type == "RealSort":
                    self.sorts[name] = RealSort()
                elif sort_type == "BoolSort":
                    self.sorts[name] = BoolSort()
                elif sort_type == "DeclareSort":
                    self.sorts[name] = DeclareSort(name)
                else:
                    raise ValueError(f"Unknown sort type: {sort_type}")
                logger.debug(f"Created sort: {name} ({sort_type})")
            except KeyError as e:
                logger.error(f"Missing required field in sort definition: {e}")
                raise ValueError(f"Invalid sort definition {sort_def}: missing {e}") from e
            except Exception as e:
                logger.error(f"Error creating sort '{name}': {e}")
                raise

    def create_functions(self) -> None:
        """Create Z3 functions from configuration.

        Raises:
            ValueError: If function definition is invalid
        """
        for func_def in self.config["functions"]:
            try:
                name = func_def["name"]
                domain = [self.sorts[sort] for sort in func_def["domain"]]
                range_sort = self.sorts[func_def["range"]]
                self.functions[name] = Function(name, *domain, range_sort)
                logger.debug(f"Created function: {name}")
            except KeyError as e:
                logger.error(f"Missing required field in function definition: {e}")
                raise ValueError(f"Invalid function definition {func_def}: missing {e}") from e
            except Exception as e:
                logger.error(f"Error creating function '{name}': {e}")
                raise

    def create_constants(self) -> None:
        """Create Z3 constants from configuration.

        Raises:
            ValueError: If constant definition is invalid
        """
        for category, constants in self.config["constants"].items():
            try:
                sort_name = constants["sort"]
                if sort_name not in self.sorts:
                    raise ValueError(f"Sort '{sort_name}' not defined")

                if isinstance(constants["members"], list):
                    # List format: ["name1", "name2"] -> create constants with those names
                    self.constants.update(
                        {c: Const(c, self.sorts[sort_name]) for c in constants["members"]}
                    )
                elif isinstance(constants["members"], dict):
                    # Dict format: {"ref_name": "z3_name"} -> create constant with z3_name, reference as ref_name
                    # FIX: Use key as both reference name AND Z3 constant name for consistency
                    self.constants.update(
                        {
                            k: Const(k, self.sorts[sort_name])
                            for k, v in constants["members"].items()
                        }
                    )
                    logger.debug(
                        "Note: Dict values in constants are deprecated, using keys as Z3 names"
                    )
                else:
                    logger.warning(f"Invalid members format for category '{category}', skipping")
                logger.debug(f"Created constants for category: {category}")
            except KeyError as e:
                logger.error(
                    f"Missing required field in constants definition for '{category}': {e}"
                )
                raise ValueError(f"Invalid constants definition: missing {e}") from e
            except Exception as e:
                logger.error(f"Error creating constants for category '{category}': {e}")
                raise

    def create_variables(self) -> None:
        """Create Z3 variables from configuration.

        Raises:
            ValueError: If variable definition is invalid
        """
        for var_def in self.config.get("variables", []):
            try:
                name = var_def["name"]
                sort_name = var_def["sort"]
                if sort_name not in self.sorts:
                    raise ValueError(f"Sort '{sort_name}' not defined")
                sort = self.sorts[sort_name]
                self.variables[name] = Const(name, sort)
                logger.debug(f"Created variable: {name} of sort {sort_name}")
            except KeyError as e:
                logger.error(f"Missing required field in variable definition: {e}")
                raise ValueError(f"Invalid variable definition {var_def}: missing {e}") from e
            except Exception as e:
                logger.error(f"Error creating variable '{name}': {e}")
                raise

    def _check_safe_ast(self, node: ast.AST, expr_str: str) -> None:
        """Check AST for dangerous constructs.

        Args:
            node: AST node to check
            expr_str: Original expression string for error messages

        Raises:
            ValueError: If dangerous construct found
        """
        for n in ast.walk(node):
            # Block attribute access to dunder methods
            if isinstance(n, ast.Attribute):
                if n.attr.startswith("__") and n.attr.endswith("__"):
                    raise ValueError(
                        f"Access to dunder attribute '{n.attr}' not allowed in '{expr_str}'"
                    )
            # Block imports
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                raise ValueError(f"Import statements not allowed in '{expr_str}'")
            # Block function/class definitions
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raise ValueError(f"Function/class definitions not allowed in '{expr_str}'")
            # Block exec/eval
            elif isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id in (
                    "eval",
                    "exec",
                    "compile",
                    "__import__",
                ):
                    raise ValueError(f"Call to '{n.func.id}' not allowed in '{expr_str}'")

    def _safe_eval(self, expr_str: str, context: dict[str, Any]) -> Any:
        """Safely evaluate expression string with restricted globals.

        Args:
            expr_str: Expression string to evaluate
            context: Local context dictionary

        Returns:
            Evaluated Z3 expression

        Raises:
            ValueError: If expression cannot be evaluated safely
        """
        # Combine Z3 operators with functions
        safe_globals = {**self.Z3_OPERATORS, **self.functions}

        try:
            # Parse to AST and check for dangerous constructs
            tree = ast.parse(expr_str, mode="eval")
            self._check_safe_ast(tree, expr_str)

            # Compile and evaluate with restricted builtins
            code = compile(tree, "<string>", "eval")
            return eval(code, {"__builtins__": {}}, {**safe_globals, **context})
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression '{expr_str}': {e}") from e
        except NameError as e:
            raise ValueError(f"Undefined name in expression '{expr_str}': {e}") from e
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expr_str}': {e}") from e

    def add_knowledge_base(self) -> None:
        """Add knowledge base assertions to solver.

        Raises:
            ValueError: If assertion is invalid
        """
        context = self.build_context()

        for assertion_entry in self.config["knowledge_base"]:
            if isinstance(assertion_entry, dict):
                assertion_str = assertion_entry["assertion"]
                value = assertion_entry.get("value", True)
            else:
                assertion_str = assertion_entry
                value = True

            try:
                expr = self._safe_eval(assertion_str, context)
                if not value:
                    expr = Not(expr)
                self.solver.add(expr)
                logger.debug(f"Added knowledge base assertion: {assertion_str[:50]}...")
            except Exception as e:
                logger.error(f"Error parsing assertion '{assertion_str}': {e}")
                raise

    def add_rules(self) -> None:
        """Add logical rules to solver.

        Raises:
            ValueError: If rule is invalid
        """
        for rule in self.config["rules"]:
            try:
                forall_vars = rule.get("forall", [])

                # Validate that if forall is specified, it's not empty
                if "forall" in rule and not forall_vars:
                    raise ValueError(
                        "Empty 'forall' list in rule - remove 'forall' key if no quantification needed"
                    )

                variables = [Const(v["name"], self.sorts[v["sort"]]) for v in forall_vars]

                if "implies" in rule:
                    if not variables:
                        raise ValueError(
                            "Implication rules require quantified variables - use 'forall' key"
                        )
                    antecedent = self.parse_expression(rule["implies"]["antecedent"], variables)
                    consequent = self.parse_expression(rule["implies"]["consequent"], variables)
                    self.solver.add(ForAll(variables, Implies(antecedent, consequent)))
                    logger.debug(f"Added implication rule with {len(variables)} variables")
                elif "constraint" in rule:
                    constraint = self.parse_expression(rule["constraint"], variables)
                    if variables:
                        self.solver.add(ForAll(variables, constraint))
                    else:
                        self.solver.add(constraint)
                    logger.debug("Added constraint rule")
                else:
                    raise ValueError(
                        f"Invalid rule (must contain 'implies' or 'constraint'): {rule}"
                    )
            except Exception as e:
                logger.error(f"Error adding rule: {e}")
                raise

    def add_verifications(self) -> None:
        """Add verification conditions.

        Raises:
            ValueError: If verification is invalid
        """
        for verification in self.config["verifications"]:
            try:
                name = verification.get("name", "unnamed_verification")

                if "exists" in verification:
                    exists_vars = verification["exists"]
                    if not exists_vars:
                        raise ValueError(f"Empty 'exists' list in verification '{name}'")
                    variables = [Const(v["name"], self.sorts[v["sort"]]) for v in exists_vars]
                    constraint = self.parse_expression(verification["constraint"], variables)
                    self.verifications[name] = Exists(variables, constraint)
                elif "forall" in verification:
                    forall_vars = verification["forall"]
                    if not forall_vars:
                        raise ValueError(f"Empty 'forall' list in verification '{name}'")
                    variables = [Const(v["name"], self.sorts[v["sort"]]) for v in forall_vars]
                    antecedent = self.parse_expression(
                        verification["implies"]["antecedent"], variables
                    )
                    consequent = self.parse_expression(
                        verification["implies"]["consequent"], variables
                    )
                    self.verifications[name] = ForAll(variables, Implies(antecedent, consequent))
                elif "constraint" in verification:
                    # Handle constraints without quantifiers
                    constraint = self.parse_expression(verification["constraint"])
                    self.verifications[name] = constraint
                else:
                    raise ValueError(
                        f"Invalid verification (must contain 'exists', 'forall', or 'constraint'): {verification}"
                    )
                logger.debug(f"Added verification: {name}")
            except Exception as e:
                logger.error(
                    f"Error processing verification '{verification.get('name', 'unknown')}': {e}"
                )
                raise

    def parse_expression(self, expr_str: str, variables: list[ExprRef] | None = None) -> ExprRef:
        """Parse expression string into Z3 expression.

        Args:
            expr_str: Expression string to parse
            variables: Optional list of quantified variables

        Returns:
            Parsed Z3 expression

        Raises:
            ValueError: If expression cannot be parsed
        """
        context = self.build_context(variables)
        return self._safe_eval(expr_str, context)

    def build_context(self, variables: list[ExprRef] | None = None) -> dict[str, Any]:
        """Build evaluation context with all defined symbols.

        Args:
            variables: Optional quantified variables to include

        Returns:
            Dictionary mapping names to Z3 objects
        """
        # Only cache context after all symbols have been loaded
        if self._context_cache is None and self._symbols_loaded:
            # Build base context once (after all sorts, functions, constants, variables loaded)
            self._context_cache = {}
            self._context_cache.update(self.functions)
            self._context_cache.update(self.constants)
            self._context_cache.update(self.variables)

        # If not cached yet, build context dynamically
        if self._context_cache is None:
            context = {}
            context.update(self.functions)
            context.update(self.constants)
            context.update(self.variables)
        else:
            context = self._context_cache.copy()

        if not variables:
            return context

        # Add quantified variables to context
        # Check for shadowing
        for v in variables:
            var_name = v.decl().name()
            if var_name in context and var_name not in [
                vv.decl().name() for vv in variables[: variables.index(v)]
            ]:
                logger.warning(f"Quantified variable '{var_name}' shadows existing symbol")
            context[var_name] = v
        return context

    def perform_actions(self) -> None:
        """Execute actions specified in configuration.

        Actions are method names to be called on this interpreter instance.
        """
        for action in self.config["actions"]:
            if hasattr(self, action):
                try:
                    logger.info(f"Executing action: {action}")
                    getattr(self, action)()
                except Exception as e:
                    logger.error(f"Error executing action '{action}': {e}")
                    raise
            else:
                logger.warning(f"Unknown action: {action}")

    def verify_conditions(self) -> None:
        """Verify all defined verification conditions.

        Checks each verification condition for satisfiability.
        Uses push/pop to isolate verification checks.

        Note: This checks satisfiability (SAT means condition can be true).
        For entailment checking (knowledge_base IMPLIES condition),
        check if knowledge_base AND NOT(condition) is UNSAT.
        """
        if not self.verifications:
            logger.info("No verifications to check")
            return

        logger.info(f"Checking {len(self.verifications)} verification condition(s)")
        self.solver.set("timeout", self.verify_timeout)

        for name, condition in self.verifications.items():
            try:
                # Use push/pop to isolate each verification check
                # This ensures verifications don't interfere with each other
                # Note: We're checking satisfiability, not entailment here
                # The condition is added AS AN ASSUMPTION to existing knowledge base
                logger.debug(f"Checking verification '{name}'")
                result = self.solver.check(condition)

                if result == sat:
                    model = self.solver.model()
                    logger.info(f"{name}: SAT")
                    logger.info(f"Model: {model}")
                elif result == unsat:
                    logger.info(f"{name}: UNSAT (condition contradicts knowledge base)")
                else:
                    logger.warning(f"{name}: UNKNOWN (timeout or incomplete)")
            except Exception as e:
                logger.error(f"Error checking verification '{name}': {e}")
                raise

    def optimize(self) -> None:
        """Run optimization if defined in configuration.

        The optimizer is separate from the solver and doesn't share constraints.
        This is intentional to allow independent optimization problems.
        """
        if "optimization" not in self.config:
            logger.info("No optimization section found.")
            return

        logger.info("Running optimization")

        try:
            # Create variables for optimization
            optimization_vars = {}
            for var_def in self.config["optimization"].get("variables", []):
                name = var_def["name"]
                sort = self.sorts[var_def["sort"]]
                optimization_vars[name] = Const(name, sort)

            # Build extended context: optimization variables + global context
            # This allows optimization constraints to reference knowledge base constants
            base_context = self.build_context()
            opt_context = {**base_context, **optimization_vars}

            # Add constraints - they can now reference both opt vars and global symbols
            for constraint in self.config["optimization"].get("constraints", []):
                # Parse with opt_var_list for quantification, but full context for evaluation
                # We need to temporarily extend context
                expr = self._safe_eval(constraint, opt_context)
                self.optimizer.add(expr)
                logger.debug(f"Added optimization constraint: {constraint[:50]}...")

            # Add objectives
            for objective in self.config["optimization"].get("objectives", []):
                expr = self._safe_eval(objective["expression"], opt_context)
                if objective["type"] == "maximize":
                    self.optimizer.maximize(expr)
                    logger.debug(f"Maximizing: {objective['expression']}")
                elif objective["type"] == "minimize":
                    self.optimizer.minimize(expr)
                    logger.debug(f"Minimizing: {objective['expression']}")
                else:
                    logger.warning(f"Unknown optimization type: {objective['type']}")

            self.optimizer.set("timeout", self.optimize_timeout)
            result = self.optimizer.check()

            if result == sat:
                model = self.optimizer.model()
                logger.info(f"Optimal Model: {model}")
            else:
                logger.warning("No optimal solution found.")
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise

    def run(self) -> None:
        """Execute the full interpretation pipeline.

        Steps:
        1. Create sorts
        2. Create functions
        3. Create constants
        4. Create variables
        5. Add knowledge base
        6. Add rules
        7. Add verifications
        8. Perform configured actions

        Raises:
            Various exceptions if any step fails
        """
        try:
            logger.info(f"Starting interpretation of {self.json_file}")
            self.create_sorts()
            self.create_functions()
            self.create_constants()
            self.create_variables()
            # Mark that all symbols have been loaded, enable caching
            self._symbols_loaded = True
            self.add_knowledge_base()
            self.add_rules()
            self.add_verifications()
            self.perform_actions()
            logger.info("Interpretation completed successfully")
        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            raise


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Z3 JSON DSL Interpreter - Execute Z3 solver configurations from JSON files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json_file", type=str, help="Path to JSON configuration file")
    parser.add_argument(
        "--verify-timeout",
        type=int,
        default=Z3JSONInterpreter.DEFAULT_VERIFY_TIMEOUT,
        help="Timeout for verification checks in milliseconds",
    )
    parser.add_argument(
        "--optimize-timeout",
        type=int,
        default=Z3JSONInterpreter.DEFAULT_OPTIMIZE_TIMEOUT,
        help="Timeout for optimization in milliseconds",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure logging when running as main script
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        interpreter = Z3JSONInterpreter(
            args.json_file,
            verify_timeout=args.verify_timeout,
            optimize_timeout=args.optimize_timeout,
        )
        interpreter.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)
