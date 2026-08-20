import os

from openai import OpenAI
import dotenv
from tqdm import tqdm

def construct_messages(
    data,
    prompt_template,
    sys_map: dict ={},
    user_map: dict ={},
):
    messages = []

    for row in data:
        system_args = {
            prompt_key: row[data_key]
            for prompt_key, data_key in sys_map.items()
        }

        user_args = {
            prompt_key: row[data_key]
            for prompt_key, data_key in user_map.items()
        }

        messages.append(
            [
                {
                    "role": "system",
                    "content": prompt_template["system_prompt"].format(
                        **system_args
                    ),
                },
                {
                    "role": "user",
                    "content": prompt_template["user_prompt"].format(
                        **user_args
                    ),
                },
            ]
        )

    return messages


def construct_messagesOLD(
    data,
    prompt_template,
    query_column,
):
    messages = []

    for row in data:
        messages.append(
            [
                {
                    "role": "system",
                    "content": prompt_template[
                        "system_prompt"
                    ],
                },
                {
                    "role": "user",
                    "content": prompt_template[
                        "user_prompt"
                    ].format(
                        question=row[
                            query_column
                        ]
                    ),
                },
            ]
        )

    return messages


def generate_openai(
    messages,
    model,
    max_tokens,
):
    dotenv.load_dotenv(
        "./resources/.env"
    )

    client = OpenAI(
        api_key=os.environ[
            "OPENAI_API_KEY"
        ],
    )

    results = []

    for prompt in tqdm(messages):
        response = client.responses.create(
            model=model,
            input=prompt,
            reasoning={
                "effort": "low",
            },
            max_output_tokens=max_tokens,
        )

        results.append(
            response.output_text.strip()
        )

    return results


def generate_vllm(
    vllm_base_url,
    messages,
    model,
    temperature,
    max_tokens,
    top_p,
):
    client = OpenAI(
        api_key="EMPTY",
        base_url=vllm_base_url,
    )

    results = []

    for prompt in tqdm(messages):
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        results.append(
            response.choices[0]
            .message.content
            .strip()
        )

    return results


def generate(
    messages,
    model,
    backend,
    vllm_base_url,
    temperature,
    max_tokens,
    top_p,
) -> list:
    assert backend in [
        "openai",
        "vllm",
    ], f"Unknown backend: {backend}"

    if backend == "openai":
        return generate_openai(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )

    if backend == "vllm":
        return generate_vllm(
            vllm_base_url=vllm_base_url,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
