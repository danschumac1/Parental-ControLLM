import json
from dataclasses import dataclass
import os
import signal
import subprocess
import time
from typing import Optional, Type, Any, List

from openai import OpenAI
from pydantic import BaseModel
from json_repair import repair_json
import requests 


@dataclass
class ChatPrompt:
    """
    Minimal chat prompt for vLLM / OpenAI-style models.
    """
    user_text: str
    system_text: Optional[str] = None
    def to_messages(self) -> list[dict[str, str]]:
        msgs = []
        if self.system_text:
            msgs.append({"role": "system", "content": self.system_text})
        msgs.append({"role": "user", "content": self.user_text})
        return msgs


class VLLMPrompter:
    """
    Simple, batched, schema-aware prompter for vLLM.
    Handles JSON repair internally.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # --------------------------------------------------
    # Core generation
    # --------------------------------------------------
    def generate(
        self,
        prompts: list[list[dict]],
    ) -> list[str]:
        """
        Fire requests back-to-back.
        vLLM will batch internally.
        """
        responses = [
            self.client.chat.completions.create(
                model=self.model,
                messages=p,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            for p in prompts
        ]
        return [
            r.choices[0].message.content.strip()
            for r in responses
        ]

    # --------------------------------------------------
    # Internal JSON handling (KEY FIX)
    # --------------------------------------------------
    def _parse_json_safely(self, text: str) -> dict:
        """
        Parse JSON from LLM output, repairing if necessary.
        """
        # First try: strict JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Second try: repair JSON-ish output
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(
                "JSON repair failed.\n"
                f"Original output:\n{text}\n"
            ) from e

    # --------------------------------------------------
    # Structured generation (schema-controlled)
    # --------------------------------------------------
    def generate_structured(
        self,
        prompts: list[list[dict]],
        schema: Type[BaseModel],
    ) -> list[Any]:
        """
        Generate outputs and parse them into a schema.
        JSON repair is applied automatically.
        """
        raw_outputs = self.generate(prompts)
        parsed: List[Any] = []

        for out in raw_outputs:
            try:
                data = self._parse_json_safely(out)

                if not isinstance(data, dict):
                    raise TypeError(
                        f"Expected JSON object, got {type(data)}"
                    )

                parsed.append(schema(**data))

            except Exception as e:
                raise ValueError(
                    f"Failed to parse output as {schema.__name__}:\n{out}"
                ) from e

        return parsed


class VllmServer:
    """
    Context manager that:
    - starts a vLLM OpenAI-compatible API server
    - waits until it is ready
    - guarantees cleanup (GPU freed) on exit, even on crashes
    """

    def __init__(
        self,
        model: str,
        cuda_device: str,
        port: int = 8000,
        host: str = "127.0.0.1",
        timeout: int = 300,
        log_file: str | None = None,
    ):
        self.model = model
        self.cuda_device = cuda_device
        self.port = port
        self.host = host
        self.timeout = timeout
        self.log_file = log_file

        self.proc: subprocess.Popen | None = None
        self._log_handle = None

    # ------------------------------------------------------------------
    # Context manager entry
    # ------------------------------------------------------------------
    def __enter__(self):
        self._start_server()
        self._wait_until_ready()
        return self

    # ------------------------------------------------------------------
    # Context manager exit (THIS IS THE IMPORTANT PART)
    # ------------------------------------------------------------------
    def __exit__(self, exc_type, exc_value, traceback):
        self._stop_server()
        # returning False propagates exceptions (good: you still see crashes)
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _start_server(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.cuda_device

        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
        ]

        stdout = stderr = subprocess.PIPE
        if self.log_file:
            self._log_handle = open(self.log_file, "w")
            stdout = stderr = self._log_handle

        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            text=True,
        )

    def _wait_until_ready(self):
        start = time.time()
        url = f"http://{self.host}:{self.port}/v1/models"

        while time.time() - start < self.timeout:
            try:
                requests.get(url, timeout=2)
                return
            except Exception:
                time.sleep(1)

        raise RuntimeError("vLLM server did not become ready")

    def _stop_server(self):
        if self.proc is None:
            return

        try:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        finally:
            if self._log_handle:
                self._log_handle.close()