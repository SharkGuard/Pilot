import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from z3dsl.knowledge_base_manager import KnowledgeBaseManager

logger = logging.getLogger(__name__)

# Initialize KnowledgeBaseManager globally or pass it around
# For simplicity, let's assume a default path for now.
# In a real application, this might be injected or configured more robustly.
kb_manager = KnowledgeBaseManager(kb_file_path="knowledge_base.json")

class UpdateKnowledgeBaseArgs(BaseModel):
    additions: List[str] = Field(..., description="List of Z3 DSL assertions to add to the knowledge base.")

class UpdateFunctionsArgs(BaseModel):
    additions: List[Dict[str, Any]] = Field(..., description="List of Z3 DSL function definitions to add.")

class UpdateConstantsArgs(BaseModel):
    additions: Dict[str, Any] = Field(..., description="Dictionary of Z3 DSL constant definitions to add.")

def update_knowledge_base(additions: List[str]) -> str:
    """
    Adds new assertions to the knowledge base after user confirmation.
    """
    if not additions:
        return "No knowledge base additions provided."

    diff = kb_manager.propose_additions(kb_additions=additions)
    print("\n--- Proposed Knowledge Base Additions (diff) ---")
    print(diff)
    print("--------------------------------------------")

    confirmation = input("Apply these knowledge base additions? (yes/no): ").lower()
    if confirmation == "yes":
        kb_manager.apply_additions(kb_additions=additions)
        return "Knowledge base updated successfully."
    else:
        return "Knowledge base additions rejected by user."

def update_functions(additions: List[Dict[str, Any]]) -> str:
    """
    Adds new function definitions after user confirmation.
    """
    if not additions:
        return "No function additions provided."

    diff = kb_manager.propose_additions(func_additions=additions)
    print("\n--- Proposed Function Additions (diff) ---")
    print(diff)
    print("--------------------------------------------")

    confirmation = input("Apply these function additions? (yes/no): ").lower()
    if confirmation == "yes":
        kb_manager.apply_additions(func_additions=additions)
        return "Functions updated successfully."
    else:
        return "Function additions rejected by user."

def update_constants(additions: Dict[str, Any]) -> str:
    """
    Adds new constant definitions after user confirmation.
    """
    if not additions:
        return "No constant additions provided."

    diff = kb_manager.propose_additions(const_additions=additions)
    print("\n--- Proposed Constant Additions (diff) ---")
    print(diff)
    print("--------------------------------------------")

    confirmation = input("Apply these constant additions? (yes/no): ").lower()
    if confirmation == "yes":
        kb_manager.apply_additions(const_additions=additions)
        return "Constants updated successfully."
    else:
        return "Constant additions rejected by user."

# Define the tools in a format compatible with LiteLLM/OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_knowledge_base",
            "description": "Add new Z3 DSL assertions to the knowledge base.",
            "parameters": UpdateKnowledgeBaseArgs.schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_functions",
            "description": "Add new Z3 DSL function definitions.",
            "parameters": UpdateFunctionsArgs.schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_constants",
            "description": "Add new Z3 DSL constant definitions.",
            "parameters": UpdateConstantsArgs.schema(),
        },
    },
]

# Map tool names to their Python functions
AVAILABLE_TOOLS = {
    "update_knowledge_base": update_knowledge_base,
    "update_functions": update_functions,
    "update_constants": update_constants,
}
