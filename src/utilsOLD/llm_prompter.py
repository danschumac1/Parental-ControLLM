import json
from typing import Any, Type

from openai import OpenAI
from pydantic import BaseModel
from json_repair import repair_json


class ChatPrompter:
    """
    OpenAI-compatible chat prompter.

    Works with:
    - OpenAI API
    - vLLM OpenAI-compatible server
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        client_kwargs = {}

        if base_url is not None:
            client_kwargs["base_url"] = base_url

        if api_key is not None:
            client_kwargs["api_key"] = api_key

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompts: list[list[dict]]) -> list[str]:
        outputs = []

        for prompt in prompts:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            outputs.append(response.choices[0].message.content.strip())

        return outputs

    def _parse_json_safely(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        try:
            repaired = repair_json(text)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(
                "JSON repair failed.\n"
                f"Original output:\n{text}\n"
            ) from e

    def generate_structured(
        self,
        prompts: list[list[dict]],
        schema: Type[BaseModel],
    ) -> list[Any]:
        raw_outputs = self.generate(prompts)
        parsed = []

        for out in raw_outputs:
            try:
                data = self._parse_json_safely(out)

                if not isinstance(data, dict):
                    raise TypeError(f"Expected JSON object, got {type(data)}")

                parsed.append(schema(**data))

            except Exception as e:
                raise ValueError(
                    f"Failed to parse output as {schema.__name__}:\n{out}"
                ) from e

        return parsed