# Z3 DSL Interpreter

A JSON-based Domain-Specific Language (DSL) for the Z3 theorem prover, providing a declarative interface for formal verification and optimization.

## Project Structure

```
proofofthought/
├── z3dsl/                      # Main package
│   ├── __init__.py             # Package exports
│   ├── interpreter.py          # Main interpreter orchestration
│   ├── cli.py                  # Command-line interface
│   │
│   ├── solvers/                # Solver abstractions
│   │   ├── __init__.py
│   │   ├── abstract.py         # AbstractSolver interface
│   │   └── z3_solver.py        # Z3Solver implementation
│   │
│   ├── security/               # Security validation
│   │   ├── __init__.py
│   │   └── validator.py        # Expression validation (AST checks)
│   │
│   ├── dsl/                    # DSL components
│   │   ├── __init__.py
│   │   ├── sorts.py            # Sort creation & topological sorting
│   │   └── expressions.py      # Expression parsing & evaluation
│   │
│   ├── verification/           # Verification logic
│   │   ├── __init__.py
│   │   └── verifier.py         # Verification condition checking
│   │
│   └── optimization/           # Optimization logic
│       ├── __init__.py
│       └── optimizer.py        # Optimization problem solving
│
├── tests/                      # Test files
│   └── 3.json                  # Example JSON configuration
│
├── run_interpreter.py          # Convenience script
├── main.py                     # Legacy monolithic version (deprecated)
└── README.md                   # This file
```

## Architecture

### Core Components

1. **Interpreter** (`interpreter.py`)
   - Orchestrates the entire interpretation pipeline
   - Coordinates between all sub-components
   - Manages configuration loading and validation

2. **Solvers** (`solvers/`)
   - `AbstractSolver`: Interface for solver implementations
   - `Z3Solver`: Z3-specific solver wrapper
   - Allows pluggable solver backends

3. **Security** (`security/`)
   - `ExpressionValidator`: AST-based security checks
   - Prevents code injection via dunder attributes, imports, eval/exec
   - Validates expressions before evaluation

4. **DSL** (`dsl/`)
   - `SortManager`: Creates and manages Z3 sorts with dependency resolution
   - `ExpressionParser`: Parses expressions with context management
   - Handles sorts, functions, constants, variables

5. **Verification** (`verification/`)
   - `Verifier`: Manages verification conditions
   - Supports ForAll, Exists, and constraint-based verification
   - Checks satisfiability with timeout support

6. **Optimization** (`optimization/`)
   - `OptimizerRunner`: Handles optimization problems
   - Supports maximize/minimize objectives
   - Separate from main solver for independent problems

## Features

### Security Enhancements
- ✅ AST-based expression validation (blocks imports, dunder access, eval/exec)
- ✅ Restricted builtin access
- ✅ Safe expression evaluation with whitelisted operators

### Correctness Fixes
- ✅ Topological sorting for sort dependencies
- ✅ Proper quantifier validation (no empty ForAll/Exists)
- ✅ Context caching with lazy initialization
- ✅ Type-safe expression parsing (returns ExprRef not BoolRef)
- ✅ BitVecSort size validation (0 < size <= 65536)

### Code Quality
- ✅ Modular architecture with separation of concerns
- ✅ Explicit imports (no wildcard imports)
- ✅ Comprehensive logging
- ✅ Type hints throughout
- ✅ Proper error handling and messages

## Usage

### Command Line

```bash
# Basic usage
python run_interpreter.py tests/3.json

# With custom timeouts
python run_interpreter.py tests/3.json \
    --verify-timeout 20000 \
    --optimize-timeout 50000

# With debug logging
python run_interpreter.py tests/3.json --log-level DEBUG
```

### As a Library

```python
from z3dsl import Z3JSONInterpreter

interpreter = Z3JSONInterpreter(
    "config.json",
    verify_timeout=10000,
    optimize_timeout=100000
)
interpreter.run()
```

### Custom Solver

```python
from z3dsl import Z3JSONInterpreter, AbstractSolver

class CustomSolver(AbstractSolver):
    # Implement abstract methods
    pass

interpreter = Z3JSONInterpreter(
    "config.json",
    solver=CustomSolver()
)
interpreter.run()
```

## JSON Configuration Format

```json
{
  "sorts": [
    {"name": "MySort", "type": "DeclareSort"},
    {"name": "MyBitVec", "type": "BitVecSort(8)"}
  ],
  "functions": [
    {"name": "f", "domain": ["MySort"], "range": "IntSort"}
  ],
  "constants": {
    "category": {
      "sort": "MySort",
      "members": ["const1", "const2"]
    }
  },
  "variables": [
    {"name": "x", "sort": "IntSort"}
  ],
  "knowledge_base": [
    "f(const1) > 0",
    {"assertion": "f(const2) < 100", "value": true}
  ],
  "rules": [
    {
      "forall": [{"name": "y", "sort": "MySort"}],
      "constraint": "f(y) >= 0"
    }
  ],
  "verifications": [
    {
      "name": "Check Property",
      "constraint": "f(const1) > f(const2)"
    }
  ],
  "actions": ["verify_conditions"]
}
```

## Dependencies

- Python 3.7+
- z3-solver

Install with:
```bash
pip install z3-solver
```

## Bug Fixes from Original

The refactored version fixes 16 critical bugs:
1. Wildcard import pollution
2. Type safety violations
3. Context cache timing issues
4. Variable shadowing
5. Security sandbox bypasses
6. Empty quantifier handling
7. Sort dependency ordering
8. Constants dict semantics
9. Optimization context isolation
10. Verification isolation
11. Logging race conditions
12. BitVecSort validation

See commit history for detailed explanations of each fix.

## Testing

```bash
# Run example test
python run_interpreter.py tests/3.json

# Expected output:
# INFO: Starting interpretation of tests/3.json
# INFO: Executing action: verify_conditions
# INFO: Checking 1 verification condition(s)
# INFO: Compare Unemployment Rates: SAT
# INFO: Model: [...]
# INFO: Interpretation completed successfully
```

## License

[Your license here]

## Contributing

Please see CONTRIBUTING.md for guidelines.
