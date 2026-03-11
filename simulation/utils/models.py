import asyncio
import os
import re
import traceback
import warnings
from datetime import datetime
from typing import Any

import pathfinder

from .logger import WandbLogger


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
            f' {message["content"]}</div>'
            for message in self.messages
        )


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

        backend_name = type(self.base_lm).__name__.lower()
        if "openrouter" in backend_name:
            self._async_client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        else:
            self._async_client = AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        self._async_client_loop = current_loop
        return self._async_client

    def _get_openai_semaphore(self):
        current_loop = asyncio.get_running_loop()
        if ModelWandbWrapper._openai_semaphore_loop is not current_loop:
            max_concurrency = int(os.getenv("OPENAI_MAX_CONCURRENCY", "8"))
            ModelWandbWrapper._openai_semaphore = asyncio.Semaphore(max_concurrency)
            ModelWandbWrapper._openai_semaphore_loop = current_loop
        return ModelWandbWrapper._openai_semaphore

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
        try:
            async with self._get_openai_semaphore():
                out = await self._get_async_client().chat.completions.create(
                    model=self.model_name,
                    messages=session.messages,
                    temperature=temperature,
                    top_p=top_p,
                    seed=self.seed,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            response_text = out.choices[0].message.content or default_value
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
        return self.base_lm

    def end_chain(self, agent_name, lm):
        html = lm.html()  # lm._html()
        html = html.replace("<s>", "")
        html = html.replace("</s>", "")
        html = html.replace("\n", "<br/>")

        def correct_rgba(html):
            """
            Corrects rgba values in the HTML string.
            """
            # This regex finds rgba values that have a decimal in the first three values
            rgba_pattern = re.compile(
                r"rgba\((\d*\.\d+|\d+), (\d*\.\d+|\d+), (\d*\.\d+|\d+),"
                r" (\d*\.\d+|\d+)\)"
            )

            def correct_rgba_match(match):
                # Correct each of the rgba values
                r, g, b, a = match.groups()
                r = int(float(r))
                g = int(float(g))
                b = int(float(b))
                return f"rgba({r}, {g}, {b}, {a})"

            # Replace incorrect rgba values with corrected ones
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
        # regex=None, TODO maybe
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
            top_p = self.top_p
        if top_p is None:
            top_p = 1.0

        try:
            lm = previous_lm + pathfinder.gen(
                name=name,
                max_tokens=max_tokens,
                stop_regex=stop_regex,
                temperature=temperature,
                top_p=top_p,
                save_stop_text=save_stop_text,
            )
            res = lm[name]
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

            # Logging
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
            top_p = self.top_p
        if top_p is None:
            top_p = 1.0

        try:
            lm = previous_lm + pathfinder.find(
                name=name,
                max_tokens=max_tokens,
                regex=regex,
                stop_regex=stop_regex,
                temperature=temperature,
                top_p=top_p,
            )
            res = lm[name]
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

            # Logging
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
        # No sampling by select, since is used more as parsing previous generated text
    ):
        start_time_ms = datetime.now().timestamp() * 1000
        prompt = previous_lm._current_prompt()
        lm = previous_lm
        res = default_value

        error_message = None
        try:
            lm = previous_lm + pathfinder.select(
                options=options,
                name=name,
            )
            res = lm[name]
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

            # Logging
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
