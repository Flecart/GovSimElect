import asyncio
import contextlib
import contextvars
import os
import re
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any

import backoff

from .logger import WandbLogger

_ACTIVE_PROMPT_ROLE = contextvars.ContextVar("active_prompt_role", default="user")


@contextlib.contextmanager
def user():
    token = _ACTIVE_PROMPT_ROLE.set("user")
    try:
        yield
    finally:
        _ACTIVE_PROMPT_ROLE.reset(token)


@contextlib.contextmanager
def assistant():
    token = _ACTIVE_PROMPT_ROLE.set("assistant")
    try:
        yield
    finally:
        _ACTIVE_PROMPT_ROLE.reset(token)


@contextlib.contextmanager
def system():
    token = _ACTIVE_PROMPT_ROLE.set("system")
    try:
        yield
    finally:
        _ACTIVE_PROMPT_ROLE.reset(token)


@dataclass
class RemoteModelSpec:
    model_name: str
    backend_name: str
    seed: int | None = None


def _resolve_backend_name(model_name: str, backend_name: str | None) -> str:
    """Auto-route non-OpenAI model ids to OpenRouter by default."""
    normalized_backend = (backend_name or "").strip()
    if not normalized_backend:
        return "OpenAI" if model_name.startswith("openai/") else "OpenRouter"

    if normalized_backend.lower() == "openai" and not model_name.startswith("openai/"):
        return "OpenRouter"

    return normalized_backend



def _normalize_remote_model_name(path: str, backend: str) -> str:
    if backend.lower() == "openai" and path.lower().startswith("openai/"):
        return path.split("/", 1)[1]
    return path

def get_model(model_name: str, seed: int | None = None, backend_name: str = "OpenAI"):
    model_name_normalized = _normalize_remote_model_name(model_name, backend_name)
    resolved_backend = _resolve_backend_name(model_name=model_name, backend_name=backend_name)
    print(f"model_name_normalized: {model_name_normalized}")
    print(f"resolved_backend: {resolved_backend}")
    return RemoteModelSpec(model_name=model_name_normalized, backend_name=resolved_backend, seed=seed)


class PromptSession:
    def __init__(self, agent_name: str, phase_name: str, query_name: str) -> None:
        self.agent_name = agent_name
        self.phase_name = phase_name
        self.query_name = query_name
        self.messages: list[dict[str, str]] = []

    def add_message(self, role: str, content: str):
        if self.messages and self.messages[-1]["role"] == role:
            self.messages[-1]["content"] += content
        else:
            self.messages.append({"role": role, "content": content})

    def add_user(self, content: str):
        self.add_message("user", content)

    def add_assistant(self, content: str):
        self.add_message("assistant", content)

    def _current_prompt(self):
        return str(self.messages)

    def html(self):
        return "".join(
            f'<div><strong>{message["role"].upper()}</strong>:'
            f' {escape(message["content"])}</div>'
            for message in self.messages
        )


class PromptChain:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.messages: list[dict[str, str]] = []
        self.variables: dict[str, Any] = {}
        self.token_in = 0
        self.token_out = 0
        self.pending_text = ""

    def add_message(self, role: str, content: str):
        if not content:
            return
        if self.messages and self.messages[-1]["role"] == role:
            self.messages[-1]["content"] += content
        else:
            self.messages.append({"role": role, "content": content})

    def __iadd__(self, content: str):
        if not isinstance(content, str):
            raise TypeError(f"Unsupported content type: {type(content)!r}")
        self.add_message(_ACTIVE_PROMPT_ROLE.get(), content)
        return self

    def __getitem__(self, key: str) -> Any:
        return self.variables[key]

    def set(self, key: str, value: Any):
        self.variables[key] = value
        return self

    def _current_prompt(self):
        return str(self.messages)

    def html(self):
        return "".join(
            f'<div><strong>{message["role"].upper()}</strong>:'
            f' {escape(message["content"])}</div>'
            for message in self.messages
        )

    def __str__(self) -> str:
        return self.html()


class ModelWandbWrapper:
    _openai_semaphore = None
    _openai_semaphore_loop = None

    def __init__(
        self,
        base_lm,
        render,
        wanbd_logger: WandbLogger,
        temperature,
        top_p,
        seed,
        is_api=False,
    ) -> None:
        self.base_lm = base_lm
        self.render = render
        self.wanbd_logger = wanbd_logger

        self.agent_chain = None
        self.chain = None
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.is_api = is_api
        self.model_name = getattr(base_lm, "model_name", None)
        self.backend_name = getattr(base_lm, "backend_name", "OpenAI")
        self._async_client = None
        self._async_client_loop = None

    def _get_async_client(self):
        current_loop = asyncio.get_running_loop()
        if (
            self._async_client is not None
            and self._async_client_loop is current_loop
        ):
            return self._async_client

        from openai import AsyncOpenAI

        backend_name = (self.backend_name or "").lower()
        if backend_name == "openrouter":
            self._async_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        elif backend_name == "openai":
            self._async_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported backend_name: {self.backend_name}")
        self._async_client_loop = current_loop
        return self._async_client

    def _get_openai_semaphore(self):
        current_loop = asyncio.get_running_loop()
        if ModelWandbWrapper._openai_semaphore_loop is not current_loop:
            max_concurrency = int(os.getenv("OPENAI_MAX_CONCURRENCY", "8"))
            ModelWandbWrapper._openai_semaphore = asyncio.Semaphore(max_concurrency)
            ModelWandbWrapper._openai_semaphore_loop = current_loop
        return ModelWandbWrapper._openai_semaphore

    def _completion_length_param(self, max_tokens: int) -> dict[str, int]:
        model_name = (self.model_name or "").lower()
        if model_name.startswith("gpt-5"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    @staticmethod
    def _reasoning_disabled_extra_body() -> dict[str, Any]:
        # qwen
        return {
            "reasoning": {"exclude": True}
        }

    async def _achat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | str | None = None,
    ) -> tuple[str, int, int]:
        async with self._get_openai_semaphore():
            length_kwargs = self._completion_length_param(max_tokens)
            out = await self._get_async_client().chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                seed=self.seed,
                stop=stop,
                **length_kwargs,
            )
        message_content = out.choices[0].message.content
        if isinstance(message_content, str):
            response_text = message_content
        elif isinstance(message_content, list):
            response_text = "".join(
                part.text for part in message_content if getattr(part, "type", None) == "text"
            )
        else:
            response_text = ""
        prompt_tokens = getattr(getattr(out, "usage", None), "prompt_tokens", 0) or 0
        completion_tokens = (
            getattr(getattr(out, "usage", None), "completion_tokens", 0) or 0
        )
        return response_text, prompt_tokens, completion_tokens

    def _complete_chain_sync(
        self,
        chain: PromptChain,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int, int]:
        return asyncio.run(
            self._achat_completion(
                chain.messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

    @staticmethod
    def _extract_prefix_and_pending(
        response_text: str,
        stop_regex: str | None,
        save_stop_text: bool,
    ) -> tuple[str, str]:
        if not stop_regex:
            return response_text, ""

        match = re.search(stop_regex, response_text)
        if match is None:
            return response_text, ""

        split_index = match.end() if save_stop_text else match.start()
        return response_text[:split_index], response_text[match.end():]

    def start_prompt(self, agent_name, phase_name, query_name) -> PromptSession:
        return PromptSession(agent_name, phase_name, query_name)

    def end_prompt(self, session: PromptSession):
        del session

    async def aclose(self) -> None:
        if self._async_client is None:
            return
        try:
            await self._async_client.close()
        finally:
            self._async_client = None
            self._async_client_loop = None

    async def acomplete_prompt(
        self,
        session: PromptSession,
        *,
        max_tokens=8000,
        temperature=None,
        top_p=None,
        stop=None,
        default_value="",
    ) -> str:
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p

        response_text = default_value

        @backoff.on_exception(backoff.expo, Exception, max_tries=5)
        async def _achat_with_backoff() -> tuple[str, int, int]:
            return await self._achat_completion(
                session.messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )

        try:
            response_text, _, _ = await _achat_with_backoff()
            response_text = response_text or default_value
        except Exception as e:
            warnings.warn(
                "An exception occured: "
                f"{e}: {traceback.format_exc()}\nReturning default value in acomplete_prompt",
                RuntimeWarning,
            )
            response_text = default_value
        finally:
            session.add_assistant(response_text)
            self.seed += 1
        return response_text

    def complete_prompt(
        self,
        session: PromptSession,
        *,
        max_tokens=8000,
        temperature=None,
        top_p=None,
        stop=None,
        default_value="",
    ) -> str:
        return asyncio.run(
            self.acomplete_prompt(
                session,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                default_value=default_value,
            )
        )

    def start_chain(
        self,
        agent_name,
        phase_name,
        query_name,
    ):
        self.agent_chain = self.wanbd_logger.get_agent_chain(agent_name, phase_name)
        self.chain = self.wanbd_logger.start_chain(phase_name + "::" + query_name)
        return PromptChain(self.model_name)

    def end_chain(self, agent_name, lm):
        html = lm.html()
        html = html.replace("<s>", "")
        html = html.replace("</s>", "")
        html = html.replace("\n", "<br/>")

        def correct_rgba(html):
            rgba_pattern = re.compile(
                r"rgba\((\d*\.\d+|\d+), (\d*\.\d+|\d+), (\d*\.\d+|\d+),"
                r" (\d*\.\d+|\d+)\)"
            )

            def correct_rgba_match(match):
                r, g, b, a = match.groups()
                r = int(float(r))
                g = int(float(g))
                b = int(float(b))
                return f"rgba({r}, {g}, {b}, {a})"

            return rgba_pattern.sub(correct_rgba_match, html)

        html = correct_rgba(html)
        self.wanbd_logger.end_chain(
            agent_name,
            self.chain,
            html,
        )

    def gen(
        self,
        previous_lm: Any,
        name=None,
        default_value="",
        *,
        max_tokens=8000,
        stop_regex=None,
        save_stop_text=False,
        temperature=None,
        top_p=None,
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()
        lm = previous_lm
        res = default_value

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p

        try:
            response_text, prompt_tokens, completion_tokens = self._complete_chain_sync(
                previous_lm,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            generated_text, pending_text = self._extract_prefix_and_pending(
                response_text,
                stop_regex,
                save_stop_text,
            )
            lm.add_message("assistant", generated_text or default_value)
            lm.pending_text = pending_text
            lm.token_in += prompt_tokens
            lm.token_out += completion_tokens
            res = generated_text or default_value
            lm.set(name, res)
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in gen",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS",
                status_message=f"valid: {True}",
                response_text=res,
                temperature=temperature,
                top_p=top_p,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm

    def find(
        self,
        previous_lm: Any,
        name=None,
        default_value="",
        *,
        max_tokens=8000,
        regex=None,
        stop_regex=None,
        temperature=None,
        top_p=None,
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()
        lm = previous_lm
        res = default_value

        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p

        try:
            search_text = lm.pending_text
            if not search_text and lm.messages and lm.messages[-1]["role"] == "assistant":
                search_text = lm.messages[-1]["content"]

            match = re.search(regex, search_text) if regex else None
            if match is not None:
                res = match.group(0)
                lm.pending_text = search_text[match.end():]
            else:
                res = default_value
                lm.pending_text = ""

            lm.add_message("assistant", res)
            lm.set(name, res)
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in find",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS",
                status_message=f"valid: {True}",
                response_text=res,
                temperature=temperature,
                top_p=top_p,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm

    def select(
        self,
        previous_lm,
        options,
        default_value=None,
        name=None,
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()
        lm = previous_lm
        res = default_value

        error_message = None
        try:
            search_text = lm.pending_text
            if not search_text and lm.messages and lm.messages[-1]["role"] == "assistant":
                search_text = lm.messages[-1]["content"]

            res = default_value
            for option in options:
                if option.lower() in search_text.lower():
                    res = option
                    break
            lm.add_message("assistant", "" if res is None else str(res))
            lm.set(name, res)
        except Exception as e:
            warnings.warn(
                f"An exception occured: {e}: {traceback.format_exc()}\nReturning default value in select",
                RuntimeWarning,
            )
            res = default_value
            lm = previous_lm.set(name, default_value)
        finally:
            if self.render:
                print(lm)
                print("-" * 20)

            end_time_ms = datetime.now().timestamp() * 1000
            self.wanbd_logger.log_trace_llm(
                chain=self.chain,
                name=name,
                default_value=default_value,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                system_message="TODO",
                prompt=prompt,
                status="SUCCESS" if error_message is None else "ERROR",
                status_message=(
                    f"valid: {True}" if error_message is None else error_message
                ),
                response_text=res,
                temperature=0.0,
                top_p=1.0,
                token_usage_in=lm.token_in,
                token_usage_out=lm.token_out,
                model_name=lm.model_name,
            )
            self.seed += 1
            return lm
