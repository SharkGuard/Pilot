"""Command-line interface for Z3 JSON DSL interpreter."""

import argparse
import logging
import sys
import os
import json # Import json for passing the program directly
from uuid import uuid4 # For generating session IDs

from examples.LLMManager import LLMManager # Import the actual LLMManager

from z3dsl.interpreter import Z3JSONInterpreter
from z3dsl.reasoning.program_generator import Z3ProgramGenerator
from z3dsl.knowledge_base_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Z3 JSON DSL Interpreter - Generate and Execute Z3 solver configurations from natural language questions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", type=str, help="Natural language question for the LLM")
    parser.add_argument(
        "--kb-file",
        type=str,
        default="knowledge_base.json",
        help="Path to the JSON knowledge base file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o", # Default model for Z3ProgramGenerator
        help="LLM model name to use for program generation",
    )
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


def main() -> None:
    """Main entry point for CLI."""
    args = parse_args()

    # Configure logging when running as main script
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    try:
        # Initialize LLM client (using a mock for now)
        # In a real application, this would be a proper client like OpenAI(api_key=...)
        # Initialize KnowledgeBaseManager first, as LLM tools might use it
        kb_manager = KnowledgeBaseManager(kb_file_path=args.kb_file)
        # Update the global kb_manager in z3dsl.llm_tools to use this instance
        # This is a workaround for the global instance in llm_tools.py
        # A more robust solution would be to pass kb_manager to Z3ProgramGenerator
        # and then to the tools, but for now, this will work.
        from z3dsl import llm_tools
        llm_tools.kb_manager = kb_manager

        # Initialize LLM client using LLMManager
        # The 'manager' argument for LLMManager is typically a higher-level orchestrator.
        # For this CLI context, we can pass None or a simple mock if it's not used.
        # session_id_ is required by LLMManager.
        llm_manager_instance = LLMManager(manager=None, session_id_=str(uuid4()), default_model_key=args.model)
        llm_client = llm_manager_instance.chat # Get the chat interface from LLMManager
        program_generator = Z3ProgramGenerator(llm_client=llm_client, model=args.model)
        

        logger.info(f"Generating Z3 DSL program for question: '{args.question}'")
        generation_result = program_generator.generate(args.question)

        if not generation_result.success or not generation_result.json_program:
            logger.error(f"Failed to generate a valid program: {generation_result.error}")
            sys.exit(1)

        # If KB was modified by tool calls, reload it to ensure the latest state
        if generation_result.kb_modified:
            logger.info("Knowledge base was modified by LLM tool calls. Reloading.")
            kb_manager = KnowledgeBaseManager(kb_file_path=args.kb_file) # Re-initialize to load latest

        # Now, run the interpreter with the generated program and the current knowledge base
        temp_json_path = "temp_generated_program.json"
        with open(temp_json_path, "w") as f:
            json.dump(generation_result.json_program, f, indent=2)
        logger.info(f"Generated program written to {temp_json_path}")

        # Merge the current knowledge base into the generated program for the interpreter
        final_program_for_interpreter = kb_manager.knowledge_base.copy()
        for key, value in generation_result.json_program.items():
            if isinstance(value, list) and key in final_program_for_interpreter and isinstance(final_program_for_interpreter[key], list):
                final_program_for_interpreter[key].extend(value)
            elif isinstance(value, dict) and key in final_program_for_interpreter and isinstance(final_program_for_interpreter[key], dict):
                final_program_for_interpreter[key].update(value)
            else:
                final_program_for_interpreter[key] = value

        temp_final_program_path = "temp_final_program_for_interpreter.json"
        with open(temp_final_program_path, "w") as f:
            json.dump(final_program_for_interpreter, f, indent=2)
        logger.info(f"Final program for interpreter (including KB) written to {temp_final_program_path}")


        interpreter = Z3JSONInterpreter(
            temp_final_program_path, # Pass the combined program
            verify_timeout=args.verify_timeout,
            optimize_timeout=args.optimize_timeout,
        )
        interpreter.run()

        # Clean up temporary files
        os.remove(temp_json_path)
        os.remove(temp_final_program_path)


    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
