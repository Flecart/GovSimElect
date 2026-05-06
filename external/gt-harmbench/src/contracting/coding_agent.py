"""Coding agent for translating NL contracts into Python enforcement code."""

import ast
import re
import time

from inspect_ai._util._async import run_coroutine
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    get_model,
    GenerateConfig,
)

from .contract import ContractMode, GameContractState
from .telemetry import record_phase_call

DEFAULT_CODING_AGENT_MODEL = "openai/gpt-5.4"
DEFAULT_CODING_AGENT_TEMPERATURE = 0.3


class ContractAgentBase:
    """Base class for contract processing agents.

    Provides common functionality for agents that process natural-language
    agreements into Python contracts.
    """

    def __init__(
        self,
        model: str,
        temperature,
    ) -> None:
        """Initialize the contract agent.

        Args:
            model: Model to use for contract processing (default: high-intelligence model).
            temperature: Temperature for generation (low for consistent output).
        """
        self.model_name = model
        self.temperature = temperature
        self.model = get_model(model)

    async def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, str, dict]:
        """Generate a response from the model.

        Args:
            system_prompt: System prompt for the model.
            user_prompt: User prompt with contract details.

        Returns:
            A tuple of (full_response, html_log, telemetry):
            - full_response: The model's completion.
            - html_log: HTML-formatted log for visualization.
            - telemetry: Token and latency metadata for the model call.
        """
        config_dict = {"temperature": self.temperature}
        started_at = time.perf_counter()
        try:
            config = GenerateConfig(temperature=self.temperature)
            response = run_coroutine(
                self.model.generate(
                    [
                        ChatMessageSystem(content=system_prompt),
                        ChatMessageUser(content=user_prompt),
                    ],
                    config=config,
                )
            )
            ended_at = time.perf_counter()
            telemetry = record_phase_call(
                phase="coding",
                role="coding_agent",
                output=response,
                started_at=started_at,
                ended_at=ended_at,
                model=self.model,
                config=config_dict,
            )
            return response.completion, "", telemetry
        except Exception as e:
            ended_at = time.perf_counter()
            telemetry = record_phase_call(
                phase="coding",
                role="coding_agent",
                started_at=started_at,
                ended_at=ended_at,
                model=self.model,
                config=config_dict,
                error=f"{type(e).__name__}: {e}",
            )
            return f"Error: {e}", f"<div class='error'>Error: {e}</div>", telemetry


class CodingAgent(ContractAgentBase):
    """Translate agreed NL contracts into Python enforcement code.

    The coding agent is a separate, high-intelligence model
    that takes the natural language specification agreed upon during
    negotiation and translates it into executable Python code.

    This follows the mech-design pattern:
    1. Models negotiate about Python code in natural language
    2. Models agree on NL specification
    3. Coding agent writes the Python code
    4. Technical validation errors are fed back for up to 3 total attempts
    """

    async def translate(
        self,
        nl_contract: str,
        mode: ContractMode,
        state: GameContractState,
        active_nl_law: str = "",
        active_code_law: str = "",
        feedback: str | None = None,
    ) -> tuple[str | None, str, str, dict]:
        """Translate an NL contract into Python enforcement code.

        Args:
            nl_contract: The agreed natural language contract.
            mode: The contracting mode (CODE_LAW only).
            state: The game state.
            active_nl_law: Previously active NL law (for amendments).
            active_code_law: Previously active Python law (for amendments).
            feedback: Technical validation feedback from a previous attempt.

        Returns:
            A tuple of (python_code, full_response, html_log, telemetry):
            - python_code: The extracted Python code, or None if translation failed.
            - full_response: The full model response for logging.
            - html_log: HTML-formatted log for visualization.
            - telemetry: Token and latency metadata for this coding-agent call.
        """
        if mode != ContractMode.CODE_LAW:
            return None, f"Unsupported coding mode: {mode.value}", "", {}

        from .prompts import coding_agent_prompt

        system_prompt, user_prompt = coding_agent_prompt(
            nl_contract=nl_contract,
            state=state,
            active_nl_law=active_nl_law,
            active_code_law=active_code_law,
            feedback=feedback,
        )

        full_response, error_html, telemetry = await self._generate_response(system_prompt, user_prompt)

        if error_html:
            return None, full_response, error_html, telemetry

        code = self._extract_python_code(full_response)

        # Build HTML log
        html_log = f"""
        <div class="coding-agent-turn">
            <h4>Coding Agent Response</h4>
            <div class="system-prompt">{system_prompt[:200]}...</div>
            <div class="response">{full_response}</div>
            <div class="extracted-code">{code or 'No code extracted'}</div>
        </div>
        """

        return code, full_response, html_log, telemetry

    def _validate_enforce_signature(self, code: str) -> tuple[bool, str | None]:
        """Validate that the code contains a properly signed enforce() function.

        Uses AST parsing to verify:
        - A function named 'enforce' exists at module level
        - It has exactly 3 parameters
        - Parameter names match expected pattern (actions, state, context)

        Args:
            code: The Python code to validate.

        Returns:
            (is_valid, error_message) - True if valid, False with error message if not.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Look for enforce function at module level
        enforce_func = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "enforce":
                enforce_func = node
                break

        if enforce_func is None:
            return False, "No 'enforce' function found at module level"

        # Check parameter count
        num_params = len(enforce_func.args.args)
        if num_params != 3:
            return False, f"enforce() must have exactly 3 parameters, found {num_params}"

        # Check parameter names (allow flexibility but warn if non-standard)
        param_names = [arg.arg for arg in enforce_func.args.args]
        standard_names = ["actions", "state", "context"]
        if param_names != standard_names:
            return False, f"enforce() parameters must be (actions, state, context), found ({', '.join(param_names)})"

        return True, None

    def _extract_python_code(self, response: str) -> str | None:
        """Extract Python code from a response.

        Looks for ```python or ``` code blocks and validates that
        the code contains a properly signed enforce() function.

        Args:
            response: The model response text.

        Returns:
            The extracted Python code, or None if not found or invalid.
        """
        patterns = [
            r"```python\s*\n?(.*?)```",
            r"```\s*\n?(.*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0].strip()

                # Validate enforce() signature using AST
                is_valid, error_msg = self._validate_enforce_signature(code)
                if is_valid:
                    return code
                # If invalid, try next pattern or return None
                # (Error is logged but not surfaced to avoid confusing the model)

        return None
