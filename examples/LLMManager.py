import logging
logger = logging.getLogger(__name__)
import litellm
import json
import os
import sys
from pydantic import BaseModel, create_model
from uuid import uuid4

class LLMManager:
    class _CompletionResponse:
        def __init__(self, content):
            self.choices = [self._Choice(content)]

        class _Choice:
            def __init__(self, content):
                self.message = self._Message(content)

            class _Message:
                def __init__(self, content):
                    self.content = content

    class _Chat:
        def __init__(self, llm_manager_instance, default_model_key: str):
            self.llm_manager = llm_manager_instance
            self.default_model_key = default_model_key

        @property
        def chat(self):
            return self

        def completions(self):
            return self._Completions(self.llm_manager, self.default_model_key)

        class _Completions:
            def __init__(self, llm_manager_instance, default_model_key: str):
                self.llm_manager = llm_manager_instance
                self.default_model_key = default_model_key

            def create(self, model, messages, max_completion_tokens, **kwargs):
                response_data = self.llm_manager.compl(
                    messages=messages,
                    model_key=self.default_model_key,
                    args={"max_tokens": max_completion_tokens, **kwargs}
                )
                
                if "error" in response_data:
                    raise Exception(response_data["error"])
                
                return LLMManager._CompletionResponse(response_data["response"])

    def __init__(self, manager, session_id_, initial_models_config: list[dict] = None, default_model_key: str = None):
        self.models = {}
        self.session_id_ = session_id_
        self.manager = manager
        self.default_model_key = default_model_key
        if default_model_key:
            self.chat = self._Chat(self, default_model_key)

        # --- Environment Variable Checks and Model Configuration Preparation ---
        configured_models = []

        # POSTGRES ENV CHECK
        # self.POSTGRES_USER = os.environ.get("POSTGRES_USER", "stylegen")
        # self.POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "stylegen")
        # self.POSTGRES_DB = os.environ.get("POSTGRES_DB", "main")
        # if self.POSTGRES_USER == "":
        #     self.error("POSTGRES environment variables not set")

        # # PINECONE_API_KEY CHECK
        # self.PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
        # if self.PINECONE_API_KEY == "":
        #     self.error("PINECONE_API_KEY environment variable not set")
        # self.INDEX_HOST = os.environ.get("INDEX_HOST", "")
        # if self.INDEX_HOST == "":
        #     self.error("INDEX_HOST environment variable not set")
        # self.NAMESPACE = os.environ.get("NAMESPACE", "__default__")

        # Define provider configurations to iterate over
        provider_configs = [
            {
                "host_type": "open_router",
                "provider_name": "openrouter",
                "env_api_key": "OPENROUTER_API_KEY",
                "env_host": "OPENROUTER_API_BASE",
                "default_host": "https://openrouter.ai/api/v1",
                "env_models_json": "OPENROUTER_MODELS_JSON",
                "env_default_model": "OPENROUTER_DEFAULT_MODEL",
                "fallback_default_model_name": "openrouter/openai/gpt-3.5-turbo",
                "api_key_required": True,
            },
            {
                "host_type": "ollama_local",
                "provider_name": "ollama",
                "env_api_key": "OLLAMA_API_KEY", # Can be empty for local Ollama
                "env_host": "OLLAMA_HOST",
                "default_host": "http://localhost:11434",
                "env_models_json": "OLLAMA_MODELS_JSON",
                "env_default_model": "OLLAMA_DEFAULT_MODEL",
                "fallback_default_model_name": "ollama/gemma3:27b",
                "api_key_required": False,
            },
            {
                "host_type": "openwebui",
                "provider_name": "openwebui",
                "env_api_key": "OPENWEBUI_API_KEY", # Can be empty
                "env_host": "OPENWEBUI_HOST",
                "default_host": "https://chat.kxsb.org/ollama",
                "env_models_json": "OPENWEBUI_MODELS_JSON",
                "env_default_model": "OPENWEBUI_DEFAULT_MODEL",
                "fallback_default_model_name": "ollama/gemma3:27b",
                "api_key_required": False,
            },
            {
                "host_type": "gemini",
                "provider_name": "gemini",
                "env_api_key": "GEMINI_API_KEY",
                "env_host": "GEMINI_HOST", # Often not needed for LiteLLM with Gemini
                "default_host": "", # No default host for Gemini, LiteLLM handles it
                "env_models_json": "GEMINI_MODELS_JSON",
                "env_default_model": "GEMINI_DEFAULT_MODEL",
                "fallback_default_model_name": "gemini/gemini-2.5-flash",
                "api_key_required": True,
            },
        ]

        for p_config in provider_configs:
            api_key = os.environ.get(p_config["env_api_key"], "")
            host_url = os.environ.get(p_config["env_host"], p_config["default_host"])
            models_json = os.environ.get(p_config["env_models_json"])

            # Skip if API key is required but missing
            if p_config["api_key_required"] and not api_key:
                continue
            
            # Process models from JSON environment variable
            if models_json:
                try:
                    provider_models = json.loads(models_json)
                    for model_cfg in provider_models:
                        # Ensure model_key is present or generated
                        if "model_key" not in model_cfg:
                            model_cfg["model_key"] = f"{p_config['host_type']}_{model_cfg['model_name'].replace('/', '_').replace('-', '_')}"
                        
                        model_cfg["host_type"] = p_config["host_type"]
                        model_cfg["host_url"] = host_url
                        model_cfg["api_key"] = api_key
                        model_cfg["provider"] = p_config["provider_name"]
                        model_cfg["tools"] = model_cfg.get("tools", True) # Default to True for tools
                        model_cfg["api_key_required"] = p_config["api_key_required"] # Add this line
                        configured_models.append(model_cfg)
                except json.JSONDecodeError:
                    self.error(f"{p_config['env_models_json']} environment variable contains invalid JSON.")
            # If no JSON, but API key (if required) and host_url (if applicable) are present, add a default model
            elif (not p_config["api_key_required"] or api_key) and (p_config["host_type"] not in ["ollama_local", "openwebui"] or host_url):
                default_model_name = os.environ.get(p_config["env_default_model"], p_config["fallback_default_model_name"])
                if default_model_name:
                    configured_models.append({
                        "model_key": f"{p_config['host_type']}_default",
                        "host_type": p_config["host_type"],
                        "host_url": host_url,
                        "api_key": api_key,
                        "provider": p_config["provider_name"],
                        "model_name": default_model_name,
                        "tools": True, # Default to True for tools
                        "api_key_required": p_config["api_key_required"] # Add this field
                    })
        
        # Add any initial models passed during instantiation
        if initial_models_config:
            # For initial_models_config, we assume api_key_required is handled externally or not strictly enforced
            # as these are manually passed. If strict validation is needed, it should be added here.
            configured_models.extend(initial_models_config)

        # Register all configured models
        for model_cfg in configured_models:
            self._register_model_from_config(model_cfg)

    def _register_model_from_config(self, model_config: dict):
        """Registers a single model configuration with LiteLLM and stores it internally.

        Args:
            model_config (dict): A dictionary containing the model's configuration.
                                 Expected keys: 'model_key', 'host_type', 'host_url',
                                 'api_key', 'provider', 'model_name', 'tools' (optional),
                                 'api_key_required' (optional), and any additional arguments.

        Examples of model_config dictionaries to be placed in JSON environment variables:
        For OPENROUTER_MODELS_JSON:
        [
            {
                "model_key": "openrouter_gpt35",
                "model_name": "openrouter/openai/gpt-3.5-turbo",
                "tools": false,
                "max_tokens": 4096
            },
            {
                "model_key": "openrouter_gemini_pro",
                "model_name": "openrouter/google/gemini-2.5-pro-preview-03-25",
                "tools": true
            }
        ]
        For OLLAMA_MODELS_JSON:
        [
            {
                "model_key": "ollama_gemma",
                "model_name": "ollama/gemma3:27b",
                "tools": true
            }
        ]
        For GEMINI_MODELS_JSON:
        [
            {
                "model_key": "gemini_pro",
                "model_name": "gemini/gemini-1.5-pro",
                "tools": true
            },
            {
                "model_key": "gemini_flash",
                "model_name": "gemini/gemini-1.5-flash",
                "tools": true
            }
        ]
        """
        model_key = model_config.get("model_key")
        host_type = model_config.get("host_type")
        host_url = model_config.get("host_url", "")
        api_key = model_config.get("api_key")
        provider = model_config.get("provider")
        model_name = model_config.get("model_name")
        tools_enabled = model_config.get("tools", False)
        api_key_required = model_config.get("api_key_required", True) # Default to True if not specified
        additional_args = {k: v for k, v in model_config.items() if k not in ["model_key", "host_type", "host_url", "api_key", "provider", "model_name", "tools", "api_key_required"]}

        # Validate essential fields, conditionally checking api_key
        if not all([model_key, host_type, provider, model_name]) or (api_key_required and not api_key):
            self.error(f"Incomplete model configuration for key '{model_key}': {model_config}")

        print(f"Registering model '{model_name}' for provider '{provider}' with key '{model_key}'")

        self.models[model_key] = {
            "api_key": api_key,
            "provider": provider,
            "base_url": host_url,
            "model": model_name,
            "tools": tools_enabled,
            **additional_args
        }

        try:
            litellm_model_params = {
                "litellm_provider": provider,
                "max_tokens": model_config.get("max_tokens", 8192), # Default max_tokens
                "api_key": api_key,
                **additional_args
            }
            
            # LiteLLM handles base_url differently for open_router vs others
            if host_url and host_type != "open_router":
                litellm_model_params["base_url"] = host_url
            elif host_type == "open_router" and host_url:
                # For open_router, base_url is typically passed directly in the completion call
                # or can be part of the model string. No global registration of base_url here.
                pass

            # Register the model with LiteLLM
            # LiteLLM expects a dict where keys are model names (e.g., "gemini/gemini-1.5-pro")
            # and values are their configurations.
            litellm.register_model({model_name: litellm_model_params})
            print(f"LiteLLM registered model: {model_name}")

        except Exception as e:
            print(f"Failed to register model '{model_name}' with LiteLLM: {e}")
            # Do not exit here, allow other models to register
            # sys.exit(1) # Removed sys.exit(1) to allow other models to be registered

    def compl(self, messages: list[dict], model_key: str, ignored_tools: list = [], args: dict = {}):
        """Performs a completion request using LiteLLM.

        Args:
            messages (list): List of message dictionaries for the LLM.
            model_key (str): The internal key for the model registered in self.models (e.g., "gemini").
                             This is NOT the LiteLLM model string, but the internal key.
            ignored_tools (list): List of tools to ignore (not currently used in this snippet).
            args (dict): Additional arguments to pass to litellm.completion.

        Returns:
            dict: A dictionary containing the LLM's response and updated messages.
        """
        if model_key not in self.models:
            self.error(f"Model with key '{model_key}' not registered in LLMManager.")

        model_config = self.models[model_key]
        model_id = model_config['model'] # This is the LiteLLM model string (e.g., "gemini/gemini-1.5-pro")
        provider = model_config['provider']
        allow_tools = model_config.get('tools', False)
        base_url = model_config.get('base_url')

        print("-----------------------------------------")
        print(f"Using model (key: {model_key}, LiteLLM ID: {model_id})")
        if base_url:
            print(f"Base URL: {base_url}")
        print("-----------------------------------------")
        
        litellm_args = {
            "model": model_id,
            "messages": messages,
            "api_key": model_config["api_key"],
            **args # Include any additional arguments passed to compl
        }

        # LiteLLM handles base_url differently for open_router vs others
        if base_url and provider == "openrouter":
             litellm_args["base_url"] = base_url
        elif base_url and provider != "openrouter":
             litellm_args["api_base"] = base_url

        # Tool calling logic (retained as per user feedback, assuming 'tools' is defined elsewhere if needed)
        # if allow_tools:
        #     # 'tools' variable needs to be defined or passed into this scope
        #     # For now, keeping it commented out to avoid NameError if 'tools' is not global
        #     # litellm_args["tools"] = tools 
        #     # litellm_args["tool_choice"] = "auto"
        
        response = litellm.completion(**litellm_args)
        response = dict(response)

        try:
            response_message = response["choices"][0].message.content
            print("------------------R-----------------------")
            print("\nLLM Response:\n")
            print(response_message)
            print("------------------R----------------------")
        except Exception as e:
            print("------------------E-----------------------")
            print(e)
            print("------------------E----------------------")
            return {"error": f"Unexpected response format from the model: {e}"}
        
        # Step 2: check if the model wanted to call a function (retained as per user feedback)
        # tool_calls =  response["choices"][0].message.tool_calls if allow_tools and "tool_calls" in response["choices"][0].message else False
        # print("\nLength of tool calls", (tool_calls))
        # if tool_calls:
        #     # Step 3: call the function
        #     # Note: the JSON response may not always be valid_; be sure to handle errors
        #     available_functions = {
        #         "get_json_element_by_id_": get_json_element_by_id,
        #         "fetch_elements_from_vector_db": fetch_elements_from_vector_db,
        #         "init_user_database":init_user_database, 
        #         "read_user_data":read_user_data, 
        #         "write_user_data":write_user_data
        #     }  
        #     messages.append(response["choices"][0].message)  # extend conversation with assistant's reply
        #     # Step 4: send the info for each function call and function response to the model
        #     for tool_call in tool_calls:
        #         function_name = tool_call.function.name
        #         function_to_call = available_functions[function_name]
        #         function_args = json.loads(tool_call.function.arguments)
        #         if function_name == "get_json_element_by_id_":
        #             function_response = function_to_call(id_=function_args.get("id_"))
        #         else:
        #             function_response = function_to_call(query=function_args.get("query"))
        #         messages.append(
        #             {
        #                 "tool_call_id_": tool_call.id_,
        #                 "role": "tool",
        #                 "name": function_name,
        #                 "content": function_response,
        #             }
        #         )  # extend conversation with function response
        
        return {"response": response_message, "messages": messages}

    def error(self, message):
        logger.error(message)
        sys.exit(1)
