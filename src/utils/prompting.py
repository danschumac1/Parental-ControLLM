import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*MatMul8bitLt: inputs will be cast.*",
)



ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / "resources" / ".env"

load_dotenv(ENV_PATH, override=True)

_TRANSFORMERS_MODELS = {}


def construct_messages(data, prompt_template, sys_map=None, user_map=None):
    sys_map = sys_map or {}
    user_map = user_map or {}

    messages = []

    for row in data:
        system_args = {prompt_key: row[data_key] for prompt_key, data_key in sys_map.items()}
        user_args = {prompt_key: row[data_key] for prompt_key, data_key in user_map.items()}

        messages.append([
            {"role": "system", "content": prompt_template["system_prompt"].format(**system_args)},
            {"role": "user", "content": prompt_template["user_prompt"].format(**user_args)},
        ])

    return messages


def generate_openai(messages, model, max_tokens):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    results = []

    for prompt in tqdm(messages):
        response = client.responses.create(model=model, input=prompt, reasoning={"effort": "low"}, max_output_tokens=max_tokens)
        results.append(response.output_text.strip())

    return results


def generate_vllm(vllm_base_url, messages, model, temperature, max_tokens, top_p):
    client = OpenAI(api_key="EMPTY", base_url=vllm_base_url)
    results = []

    for prompt in tqdm(messages):
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
                extra_body={
                    "chat_template_kwargs": {
                        "enable_thinking": False
                    }
                }

        )

        results.append(response.choices[0].message.content.strip())

    return results


_TRANSFORMERS_MODELS = {}


def load_transformers_model(model):
    if model in _TRANSFORMERS_MODELS:
        return _TRANSFORMERS_MODELS[model]

    import torch
    from transformers import (
        AutoModelForMultimodalLM,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    print(f"Loading Transformers model: {model}", flush=True)

    processor = AutoProcessor.from_pretrained(model)

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    hf_model = AutoModelForMultimodalLM.from_pretrained(
        model,
        device_map="auto",
        quantization_config=quantization_config,
    )

    hf_model.eval()

    _TRANSFORMERS_MODELS[model] = (hf_model, processor)

    print(
        f"Loaded {model} on {next(hf_model.parameters()).device}",
        flush=True,
    )

    print(
        f"Model footprint: "
        f"{hf_model.get_memory_footprint() / 1024**3:.2f} GiB",
        flush=True,
    )

    return hf_model, processor


def generate_transformers(
    messages,
    model,
    temperature,
    max_tokens,
    top_p,
):
    import torch

    hf_model, processor = load_transformers_model(model)

    results = []

    for i, prompt in enumerate(
        tqdm(messages, desc="Generating"),
        start=1,
    ):
        inputs = processor.apply_chat_template(
            prompt,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(hf_model.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        input_length = inputs["input_ids"].shape[-1]

        print(
            f"\nPrompt {i}/{len(messages)} "
            f"| tokens={input_length:,}",
            flush=True,
        )

        generation_kwargs = {
            "max_new_tokens": max_tokens,
        }

        if temperature == 0.0:
            generation_kwargs["do_sample"] = False
        else:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                }
            )

        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            outputs = hf_model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_tokens = outputs[0, input_length:]

        result = processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        results.append(result)

        print(
            f"GPU allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB | "
            f"reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB | "
            f"peak: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )

        del inputs
        del outputs
        del generated_tokens

    return results


def generate(
    messages,
    model,
    backend,
    vllm_base_url=None,
    temperature=0.0,
    max_tokens=1024,
    top_p=1.0,
) -> list[str]:

    if backend == "openai":
        return generate_openai(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )

    elif backend == "vllm":
        if vllm_base_url is None:
            raise ValueError("vllm_base_url is required for the vLLM backend.")

        return generate_vllm(
            vllm_base_url=vllm_base_url,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    elif backend == "transformers":
        return generate_transformers(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

    else:
        raise ValueError(f"Unknown backend: {backend}")