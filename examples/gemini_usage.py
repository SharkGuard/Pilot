#!/usr/bin/env python3
"""Example: Simple usage of ProofOfThought API with Gemini."""

import logging
import os

from LLMManager import LLMManager
from z3dsl.reasoning import ProofOfThought

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Gemini Configuration using LLMManager ---
# Dummy manager and session_id for LLMManager initialization
class DummyManager:
    pass

dummy_manager = DummyManager()
dummy_session_id = "test_session"

# Define Gemini model configuration
# This can also be loaded from GEMINI_MODELS_JSON environment variable
gemini_model_config = [
    {
        "model_key": "gemini_pro_local", # A unique key for internal use
        "host_type": "gemini",
        "host_url": os.getenv("GEMINI_HOST", ""), # Can be empty
        "api_key": os.getenv("GEMINI_API_KEY"),
        "provider": "gemini",
        "model_name": "gemini/gemini-2.5-flash", # LiteLLM model string
        "tools": True,
        "max_tokens": 8192
    }
]

# Initialize LLMManager with the Gemini model configuration and default model key
llm_manager = LLMManager(manager=dummy_manager, session_id_=dummy_session_id, initial_models_config=gemini_model_config, default_model_key="gemini_pro_local")

# Initialize ProofOfThought directly with llm_manager.chat
pot = ProofOfThought(llm_client=llm_manager.chat, model="gemini/gemini-2.5-flash")

# --- End Gemini Configuration ---

# Ask a question
question = "give me a good network security policy for a small business"
result = pot.query(question)

# Print results
print("\n" + "=" * 80)
print("QUERY RESULTS")
print("=" * 80)
print(f"Question: {result.question}")
print(f"Answer: {result.answer}")
print(f"Success: {result.success}")
print(f"Attempts: {result.num_attempts}")
print(f"SAT count: {result.sat_count}")
print(f"UNSAT count: {result.unsat_count}")

if result.error:
    print(f"Error: {result.error}")

if result.json_program:
    print("\nGenerated JSON program structure:")
    print(f"  - Sorts: {len(result.json_program.get('sorts', []))}")
    print(f"  - Functions: {len(result.json_program.get('functions', []))}")
    print(f"  - Constants: {len(result.json_program.get('constants', {}))}")
    print(f"  - Knowledge base: {len(result.json_program.get('knowledge_base', []))}")
    print(f"  - Verifications: {len(result.json_program.get('verifications', []))}")
