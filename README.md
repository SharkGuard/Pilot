# SharkGuard Pilot

LLM-driven Z3 DSL program generation and interactive theorem proving.

SharkGuard Pilot is an advanced system that leverages Large Language Models (LLMs) to translate natural language questions into formal Z3 DSL (Domain-Specific Language) programs. It then uses the Z3 theorem prover to answer complex reasoning tasks. 
> - ***This project introduces interactive knowledge base management, allowing LLMs to propose extensions to the system's knowledge base, functions, and constants, with explicit user confirmation and diff previews.***
> - pls see https://github.com/SharkGuard/Pilot/issues/1 for current status
> - rest of the readme is mostly a placeholder and bullshit
## Features

*   **LLM-Driven Program Generation**: Convert natural language questions into structured Z3 DSL JSON programs.
*   **Controlled LLM Generation**: Strict guidelines ensure the LLM only generates knowledge base elements (functions, constants, assertions) that are explicitly provided or directly implied by the user's query.
*   **Interactive Knowledge Base Management**: LLMs can propose additions to the knowledge base, functions, and constants via tool calls.
*   **User Confirmation with Diff Previews**: All proposed knowledge base changes are presented to the user with a `git diff`-like preview, requiring explicit confirmation before application.
*   **Z3 Theorem Proving**: Utilizes the powerful Z3 solver for verification and optimization of logical programs.
*   **Extensible Architecture**: Designed with modularity to support various LLM providers and future enhancements.

## Quick Start

To use SharkGuard Pilot, you will typically interact via the command-line interface, providing a natural language question. The system will generate a Z3 DSL program, potentially interactively updating its knowledge base.

First, ensure your LLM client (e.g., OpenAI, Gemini via LiteLLM) is configured, typically through environment variables for API keys.

```bash
# Example: Run a query
python3 run_interpreter.py "Is it true that all birds can fly?"
```

If the LLM proposes changes to the knowledge base, you will see a diff and be prompted for confirmation:

```
--- Proposed Knowledge Base Changes (diff) ---
--- current_knowledge_base.json
+++ proposed_knowledge_base.json
@@ -1,5 +1,10 @@
 {
   "knowledge_base": [
     "is_bird(sparrow)",
     "can_fly(sparrow)"
   ]
+  "functions": [
+    {
+      "name": "is_bird",
+      "domain": ["Animal"],
+      "range": "BoolSort"
+    }
+  ]
 }
--------------------------------------------
Apply these changes to the knowledge base? (yes/no): yes
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SharkGuard/Pilot.git
    cd Pilot
    ```

2.  **Install dependencies:**
    ```bash
    pip install z3-solver openai litellm pydantic scikit-learn numpy
    ```
    (Note: `openai` is needed if you use OpenAI models directly, `litellm` for broader LLM support, `pydantic` for tool schemas.)

3.  **Configure LLM API Keys**:
    Set your LLM API keys as environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_HOST`, etc., depending on your `LLMManager` configuration). Refer to `examples/LLMManager.py` for details on supported environment variables.

## Architecture

SharkGuard Pilot operates with a layered architecture:

1.  **High-level API (`z3dsl.reasoning`)**: Provides a simplified Python interface for reasoning tasks, abstracting away the complexities of DSL generation and solver interaction.
2.  **LLM Program Generation (`z3dsl.reasoning.program_generator`)**: Interacts with LLMs to generate Z3 DSL programs from natural language. This layer is responsible for prompt engineering, handling multi-turn conversations, and processing LLM tool calls.
3.  **LLM Tools (`z3dsl.llm_tools`)**: Defines specific functions (e.g., `update_knowledge_base`, `update_functions`, `update_constants`) that the LLM can call to propose modifications to the system's knowledge base. These tools include user confirmation and diff generation.
4.  **Knowledge Base Management (`z3dsl.knowledge_base_manager`)**: Manages the persistent JSON-based knowledge base, including loading, saving, and applying confirmed additions.
5.  **Low-level DSL (`z3dsl`)**: The core interpreter and Z3 theorem prover interface, responsible for parsing the JSON DSL into Z3 expressions and executing solver configurations.
6.  **Command-Line Interface (`z3dsl.cli`)**: The main entry point for users, orchestrating the entire flow from natural language input to Z3 result, including interactive knowledge base updates.

## Examples

See the `examples/` directory for complete examples, including various LLM configurations and usage patterns.
