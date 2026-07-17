"""
Prompter utilities for OpenAI API models and local vLLM offline inference.

Expected prompt format:
    prompts: list[list[dict]]

Example:
    [
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write a JSON object."}
        ],
        ...
    ]
"""

# Standard library imports
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
from typing import Any, Optional

# Third-party imports
from json_repair import repair_json
from openai import OpenAI
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from dotenv import load_dotenv

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


class BasePrompter(ABC):
    """
    Abstract base class for all prompters.

    Child classes only need to implement generate().
    JSON repair and structured parsing are shared.
    """

    @abstractmethod
    def generate(self, prompts: list[list[dict]]) -> list[str]:
        """
        Generate raw text outputs from a list of chat-style prompts.

        Args:
            prompts:
                A list of prompts, where each prompt is a list of chat messages.

        Returns:
            A list of generated strings, one per prompt.
        """
        pass

    def _parse_json_safely(self, text: str) -> dict:
        """
        Parse JSON from model output.

        First tries strict JSON parsing.
        If that fails, tries json_repair.
        """

        # First try: valid JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Second try: repair malformed JSON
        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(
                "JSON repair failed.\n"
                f"Original output:\n{text}"
            ) from e

    def generate_structured(
        self,
        prompts: list[list[dict]],
        schema: type[BaseModel],
    ) -> list[Any]:
        """
        Generate outputs and parse each output into a Pydantic schema.

        Args:
            prompts:
                A list of chat-style prompts.

            schema:
                A Pydantic BaseModel class.

        Returns:
            A list of parsed schema instances.
        """

        raw_outputs = self.generate(prompts)
        parsed: list[Any] = []

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


class VLLMPrompter(BasePrompter):
    """
    Prompter for local vLLM offline inference.

    This does not use `vllm serve`.
    It loads the model directly in Python and calls llm.generate()
    on the full batch of prompts.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
    ):
        load_dotenv("./resources/.env")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        self.llm = LLM(
            model=self.model,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert OpenAI-style chat messages into a model-specific prompt string.

        This is important for instruct/chat models such as Qwen, Llama, Mistral,
        etc., because they expect their own chat template format.
        """

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        """
        Generate outputs using vLLM offline inference.

        vLLM receives the full list of prompt strings and handles batching.
        """

        prompt_texts = [
            self._messages_to_prompt(messages)
            for messages in prompts
        ]

        outputs = self.llm.generate(
            prompt_texts,
            self.sampling_params,
        )

        return [
            output.outputs[0].text.strip()
            for output in outputs
        ]


class OpenAIPrompter(BasePrompter):
    """
    Prompter for OpenAI API models.

    Uses OPENAI_API_KEY from the environment.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        load_dotenv("./resources/.env")
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        """
        Generate outputs using the OpenAI chat completions API.

        This is intentionally simple and sequential.
        """

        outputs: list[str] = []

        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
            )

            outputs.append(
                response.choices[0].message.content.strip()
            )

        return outputs