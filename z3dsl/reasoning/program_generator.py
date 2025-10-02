"""Z3 DSL program generator using LLM."""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from z3dsl.reasoning.prompt_template import build_prompt

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of JSON program generation."""

    json_program: dict[str, Any] | None
    raw_response: str
    success: bool
    error: str | None = None


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

            # Make LLM API call (compatible with both OpenAI and Azure OpenAI)
            # Azure OpenAI requires content as string, not list
            # GPT-5 only supports temperature=1 (default), so don't pass it
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content

            # Extract JSON from markdown code block
            json_program = self._extract_json(raw_response)

            if json_program:
                return GenerationResult(
                    json_program=json_program,
                    raw_response=raw_response,
                    success=True,
                )
            else:
                # Log the raw response to help debug extraction failures
                logger.debug(f"Raw LLM response:\n{raw_response[:1000]}...")
                return GenerationResult(
                    json_program=None,
                    raw_response=raw_response,
                    success=False,
                    error="Failed to extract valid JSON from response",
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
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": previous_response},
                    {"role": "user", "content": feedback_message},
                ],
                max_completion_tokens=max_tokens,
            )

            raw_response = response.choices[0].message.content
            json_program = self._extract_json(raw_response)

            if json_program:
                return GenerationResult(
                    json_program=json_program,
                    raw_response=raw_response,
                    success=True,
                )
            else:
                # Log the raw response to help debug extraction failures
                logger.debug(f"Raw LLM feedback response:\n{raw_response[:1000]}...")
                return GenerationResult(
                    json_program=None,
                    raw_response=raw_response,
                    success=False,
                    error="Failed to extract valid JSON from feedback response",
                )

        except Exception as e:
            logger.error(f"Error generating program with feedback: {e}")
            return GenerationResult(
                json_program=None,
                raw_response="",
                success=False,
                error=str(e),
            )

    def _extract_json(self, markdown_content: str) -> dict[str, Any] | None:
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
