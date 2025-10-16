"""Z3 DSL program generator using LLM."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from z3dsl.reasoning.prompt_template import build_prompt
from z3dsl.llm_tools import TOOLS, AVAILABLE_TOOLS # Import tools

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of JSON program generation."""

    json_program: Optional[dict[str, Any]]
    raw_response: str
    success: bool
    error: Optional[str] = None
    kb_modified: bool = False # New field to indicate if KB was modified by tools


class Z3ProgramGenerator:
    """Generate Z3 DSL JSON programs from natural language questions using LLM."""

    def __init__(self, llm_client: Any, model: str = "gpt-4o") -> None:
        """Initialize the program generator.

        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            model: Model name to use
        """
        self.llm_client = llm_client
        self.model = model

    def generate(
        self,
        question: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
    ) -> GenerationResult:
        """Generate a Z3 DSL JSON program from a question.

        Args:
            question: Natural language question
            temperature: LLM temperature
            max_tokens: Maximum tokens for response (default 16384 for GPT-5)

        Returns:
            GenerationResult with JSON program or error
        """
        try:
            prompt = build_prompt(question)
            messages = [{"role": "user", "content": prompt}]
            kb_modified = False
            tool_call_attempts = 0
            MAX_TOOL_CALL_ATTEMPTS = 5 # Limit the number of consecutive tool calls

            while tool_call_attempts < MAX_TOOL_CALL_ATTEMPTS:
                response = self.llm_client.chat.completions().create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    tools=TOOLS, # Enable tool calling
                    tool_choice="auto", # Allow LLM to decide whether to call a tool
                )

                response_message = response.choices[0].message
                raw_response = response_message.content or "" # Store raw response for debugging

                tool_calls = response_message.tool_calls
                if tool_calls:
                    messages.append(response_message) # Extend conversation with assistant's reply
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = AVAILABLE_TOOLS.get(function_name)
                        if function_to_call:
                            try:
                                function_args = json.loads(tool_call.function.arguments)
                                logger.info(f"Calling tool: {function_name} with args: {function_args}")
                                tool_response = function_to_call(**function_args)
                                logger.info(f"Tool {function_name} response: {tool_response}")
                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": tool_response,
                                    }
                                )
                                kb_modified = True # Assume KB is modified if tool is called
                            except Exception as tool_e:
                                error_msg = f"Error executing tool {function_name}: {tool_e}"
                                logger.error(error_msg)
                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": f"Error: {error_msg}",
                                    }
                                )
                        else:
                            error_msg = f"LLM attempted to call unknown tool: {function_name}"
                            logger.error(error_msg)
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": f"Error: {error_msg}",
                                }
                            )
                    tool_call_attempts += 1
                elif raw_response:
                    # LLM provided a content message, extract JSON
                    json_program = self._extract_json(raw_response)
                    if json_program:
                        return GenerationResult(
                            json_program=json_program,
                            raw_response=raw_response,
                            success=True,
                            kb_modified=kb_modified,
                        )
                    else:
                        logger.debug(f"Raw LLM response:\n{raw_response[:1000]}...")
                        return GenerationResult(
                            json_program=None,
                            raw_response=raw_response,
                            success=False,
                            error="Failed to extract valid JSON from response",
                            kb_modified=kb_modified,
                        )
                else:
                    # No tool calls and no content, something went wrong
                    return GenerationResult(
                        json_program=None,
                        raw_response=raw_response,
                        success=False,
                        error="LLM returned an empty response or neither tool call nor content.",
                        kb_modified=kb_modified,
                    )
            
            # If loop exits due to max tool call attempts
            return GenerationResult(
                json_program=None,
                raw_response=raw_response,
                success=False,
                error=f"Exceeded maximum tool call attempts ({MAX_TOOL_CALL_ATTEMPTS}).",
                kb_modified=kb_modified,
            )

        except Exception as e:
            logger.error(f"Error generating program: {e}")
            return GenerationResult(
                json_program=None,
                raw_response="",
                success=False,
                error=str(e),
            )

    def generate_with_feedback(
        self,
        question: str,
        error_trace: str,
        previous_response: str,
        temperature: float = 0.1,
        max_tokens: int = 16384,
    ) -> GenerationResult:
        """Regenerate program with error feedback.

        Args:
            question: Original question
            error_trace: Error message from previous attempt
            previous_response: Previous LLM response
            temperature: LLM temperature
            max_tokens: Maximum tokens (default 16384 for GPT-5)

        Returns:
            GenerationResult with corrected JSON program
        """
        try:
            prompt = build_prompt(question)
            feedback_message = (
                f"There was an error processing your response:\n{error_trace}\n"
                "Please fix the JSON accordingly."
            )

            # Multi-turn conversation with error feedback
            # Compatible with both OpenAI and Azure OpenAI
            # GPT-5 only supports temperature=1 (default), so don't pass it
            response = self.llm_client.chat.completions().create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": previous_response},
                    {"role": "user", "content": feedback_message},
                ],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content
            prompt = build_prompt(question)
            feedback_message = (
                f"There was an error processing your response:\n{error_trace}\n"
                "Please fix the JSON accordingly."
            )
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": previous_response},
                {"role": "user", "content": feedback_message},
            ]
            kb_modified = False # Assume KB not modified initially in feedback loop
            tool_call_attempts = 0
            MAX_TOOL_CALL_ATTEMPTS = 5

            while tool_call_attempts < MAX_TOOL_CALL_ATTEMPTS:
                response = self.llm_client.chat.completions().create(
                    model=self.model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    tools=TOOLS, # Enable tool calling
                    tool_choice="auto", # Allow LLM to decide whether to call a tool
                )

                response_message = response.choices[0].message
                raw_response = response_message.content or ""

                tool_calls = response_message.tool_calls
                if tool_calls:
                    messages.append(response_message)
                    for tool_call in tool_calls:
                        function_name = tool_call.function.name
                        function_to_call = AVAILABLE_TOOLS.get(function_name)
                        if function_to_call:
                            try:
                                function_args = json.loads(tool_call.function.arguments)
                                logger.info(f"Calling tool: {function_name} with args: {function_args}")
                                tool_response = function_to_call(**function_args)
                                logger.info(f"Tool {function_name} response: {tool_response}")
                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": tool_response,
                                    }
                                )
                                kb_modified = True
                            except Exception as tool_e:
                                error_msg = f"Error executing tool {function_name}: {tool_e}"
                                logger.error(error_msg)
                                messages.append(
                                    {
                                        "tool_call_id": tool_call.id,
                                        "role": "tool",
                                        "name": function_name,
                                        "content": f"Error: {error_msg}",
                                    }
                                )
                        else:
                            error_msg = f"LLM attempted to call unknown tool: {function_name}"
                            logger.error(error_msg)
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": f"Error: {error_msg}",
                                }
                            )
                    tool_call_attempts += 1
                elif raw_response:
                    json_program = self._extract_json(raw_response)
                    if json_program:
                        return GenerationResult(
                            json_program=json_program,
                            raw_response=raw_response,
                            success=True,
                            kb_modified=kb_modified,
                        )
                    else:
                        logger.debug(f"Raw LLM feedback response:\n{raw_response[:1000]}...")
                        return GenerationResult(
                            json_program=None,
                            raw_response=raw_response,
                            success=False,
                            error="Failed to extract valid JSON from feedback response",
                            kb_modified=kb_modified,
                        )
                else:
                    return GenerationResult(
                        json_program=None,
                        raw_response=raw_response,
                        success=False,
                        error="LLM returned an empty response or neither tool call nor content in feedback.",
                        kb_modified=kb_modified,
                    )
            
            return GenerationResult(
                json_program=None,
                raw_response=raw_response,
                success=False,
                error=f"Exceeded maximum tool call attempts ({MAX_TOOL_CALL_ATTEMPTS}) in feedback loop.",
                kb_modified=kb_modified,
            )

        except Exception as e:
            logger.error(f"Error generating program with feedback: {e}")
            return GenerationResult(
                json_program=None,
                raw_response="",
                success=False,
                error=str(e),
            )

    def _extract_json(self, markdown_content: str) -> Optional[dict[str, Any]]:
        """Extract JSON from markdown code block.

        Args:
            markdown_content: Markdown text potentially containing JSON

        Returns:
            Parsed JSON dict or None if extraction failed
        """
        # Pattern to match ```json ... ``` code blocks
        json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
        match = re.search(json_pattern, markdown_content)

        if match:
            try:
                json_str = match.group(1)
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                return None

        # Try to find JSON without code block markers
        try:
            # Look for { ... } pattern
            brace_pattern = r"\{[\s\S]*\}"
            match = re.search(brace_pattern, markdown_content)
            if match:
                return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

        return None
