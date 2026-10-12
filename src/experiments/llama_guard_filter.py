"""Classify student queries with Llama Guard 4, full HECAT policies, and Transformers.

Run from the project root:
    CUDA_VISIBLE_DEVICES=0 python ./src/experiments/llama_guard_filter.py \
        --input_data_path ./data/generated/sample.tsv --n_rows 3 --save_every 1

Uses the verified Transformers workaround: attention_chunk_size=8192,
SDPA attention, typed text content blocks, and use_cache=False.
"""
import sys; sys.path.append('./src')
import os
import json
import time
import hashlib
import argparse
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForMultimodalLM
from utils.file_io import load_tsv_file, load_single_yaml_prompt, append_jsonl_file, save_args_as_config_json
from utils.evaluation import evaluate_dataset, print_evaluation, save_evaluation, parse_bool

TEKS_FILE = 'data/cleaned/hecat_standards.tsv'
PROMPT_PATH = 'data/prompts/llama_guard/hecat_expectations.yaml'
OUTPUT_DIR = 'data/generated/llama_guard_results'
ALL_CATS = {
    'AOD': 'Alcohol and Other Drug-Use Prevention',
    'FN': 'Food and Nutrition',
    'MEH': 'Mental and Emotional Health',
    'PA': 'Physical Activity',
    'PHW': 'Personal Health and Wellness',
    'S': 'Safety',
    'SH': 'Sexual Health',
    'T': 'Tobacco-Use Prevention',
    'V': 'Violence Prevention',
}
REFUSAL = "Sorry, I can't help with that."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--input_data_path', required=True)
    p.add_argument('--model', default='meta-llama/Llama-Guard-4-12B')
    p.add_argument('--backend', choices=['transformers'], default='transformers')
    p.add_argument('--query_column', default='situation')
    p.add_argument('--prompt_path', default=PROMPT_PATH)
    p.add_argument('--n_rows', type=int, default=-1)
    p.add_argument('--save_every', type=int, default=5, help='Save frequency; inference is sequential to conserve VRAM')
    p.add_argument('--max_tokens', type=int, default=32)
    p.add_argument('--attention_chunk_size', type=int, default=8192)
    p.add_argument('--save_prompts', action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def build_prompt(row: dict, teks_df: pd.DataFrame, template: dict, query_column: str) -> tuple[str, list[str]]:
    pretend_age = int(row['pretend_age'])
    blocked = {x.strip() for x in str(row.get('blocked_cats', '') or '').split(',') if x.strip()}
    unknown = blocked - ALL_CATS.keys()
    if unknown:
        raise ValueError(f'Unknown blocked categories: {unknown}')
    disallowed = teks_df.loc[
        teks_df['module_code'].isin(blocked) | (teks_df['grade_code'] > pretend_age)
    ]
    parts = []
    for code, group in disallowed.groupby('module_code', sort=True):
        expectations = '\n'.join(f'- {x}' for x in group['expectation'].drop_duplicates())
        parts.append(f'{code}: {ALL_CATS[code]}.\nRestricted expectations:\n{expectations}')
    text = '\n\n'.join(parts) if parts else 'None'
    prompt = template['prompt'].format(disallowed_teks=text, question=row[query_column])
    return prompt, sorted(disallowed['module_code'].unique().tolist())


def parse_guard_output(generation: str, offered_categories: list[str]) -> tuple[bool, list[str]]:
    lines = [line.strip() for line in generation.strip().splitlines() if line.strip()]
    if not lines or lines[0].lower() not in ('safe', 'unsafe'):
        raise ValueError(f'Unexpected Llama Guard output: {generation!r}')
    unsafe = lines[0].lower() == 'unsafe'
    if not unsafe:
        if len(lines) != 1:
            raise ValueError(f'Unexpected extra text after safe: {generation!r}')
        return False, []
    if len(lines) != 2:
        raise ValueError(f'Unsafe output must include category codes: {generation!r}')
    cats = [x.strip().upper() for x in lines[1].split(',')]
    if not cats or any(x not in offered_categories for x in cats):
        raise ValueError(f'Invalid categories in Llama Guard output: {generation!r}')
    return True, sorted(set(cats))


def load_existing_outputs(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    outputs = {}
    with path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'id' not in row or row['id'] in outputs:
                raise ValueError(f'Missing/duplicate ID in {path}, line {line_num}')
            outputs[row['id']] = row
    return outputs


def score_outputs(rows: list[dict]) -> dict:
    proxy = [{**row, 'generation': REFUSAL if row['refusal_pred'] else 'safe'} for row in rows]
    return evaluate_dataset(proxy)


def load_model(model_name: str, attention_chunk_size: int):
    print('Loading processor and configuration...', flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    print(f'Original attention_chunk_size: {getattr(config.text_config, "attention_chunk_size", None)}', flush=True)
    config.text_config.attention_chunk_size = attention_chunk_size
    print('Loading model once (BF16, SDPA)...', flush=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name, config=config, dtype=torch.bfloat16,
        device_map='auto', attn_implementation='sdpa'
    )
    model.eval()
    return processor, model


def classify(prompt: str, processor, model, max_tokens: int) -> tuple[str, int, float]:
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors='pt'
    ).to(model.device)
    n_tokens = inputs['input_ids'].shape[-1]
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=False)
    elapsed = time.perf_counter() - start
    output = processor.decode(outputs[0][n_tokens:], skip_special_tokens=True)
    del outputs, inputs
    return output, n_tokens, elapsed


def main():
    args = parse_args()
    if args.save_every < 1 or args.max_tokens < 1 or args.attention_chunk_size < 1:
        raise ValueError('--save_every, --max_tokens, and --attention_chunk_size must be positive')
    data = load_tsv_file(args.input_data_path)
    if args.n_rows > 0:
        data = data[:args.n_rows]
    if not data:
        print('No rows found.')
        return
    required = {'id', args.query_column, 'pretend_age', 'blocked_cats', 'refusal_gt'}
    for i, row in enumerate(data, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Row {i} missing fields: {missing}')
    ids = [row['id'] for row in data]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate input IDs')

    teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    teks_df['grade_code'] = pd.to_numeric(teks_df['grade_code'])
    teks_df['module_code'] = teks_df['module_code'].astype(str).str.strip()
    if set(teks_df['module_code']) - ALL_CATS.keys():
        raise ValueError('Unknown HECAT module(s)')
    template = load_single_yaml_prompt(args.prompt_path)
    if 'prompt' not in template:
        raise ValueError(f'Prompt YAML must contain a "prompt" field: {args.prompt_path}')

    safe_model = args.model.replace('/', '--')
    dataset_name = Path(args.input_data_path).stem
    run_dir = Path(OUTPUT_DIR) / safe_model / dataset_name / args.query_column / 'transformers'
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / 'outputs.jsonl'
    eval_path = run_dir / 'eval.json'
    prompt_dir = run_dir / 'prompts'
    existing = load_existing_outputs(output_path)
    selected = set(ids)
    completed = {k: v for k, v in existing.items() if k in selected}
    remaining = [r for r in data if r['id'] not in completed]
    if existing and any('refusal_pred' not in r for r in completed.values()):
        raise ValueError('Existing output is missing classifier results')
    if not existing:
        save_args_as_config_json(args, str(output_path))
    print(f'Loaded: {len(data)} | Completed: {len(completed)} | Remaining: {len(remaining)}', flush=True)

    processor = model = None
    if any(build_prompt(row, teks_df, template, args.query_column)[1] for row in remaining):
        processor, model = load_model(args.model, args.attention_chunk_size)

    for start in range(0, len(remaining), args.save_every):
        batch = remaining[start:start + args.save_every]
        batch_output = []
        for row in batch:
            prompt, offered = build_prompt(row, teks_df, template, args.query_column)
            prompt_path = None
            if args.save_prompts:
                prompt_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(str(row['id']).encode('utf-8')).hexdigest()[:16]
                prompt_path = prompt_dir / f'{digest}.txt'
                prompt_path.write_text(prompt, encoding='utf-8')
            if offered:
                generation, prompt_tokens, latency = classify(prompt, processor, model, args.max_tokens)
                try:
                    refusal_pred, violated = parse_guard_output(generation, offered)
                except ValueError:
                    if prompt_path:
                        failure = prompt_path.with_suffix('.response.txt')
                        failure.write_text(generation, encoding='utf-8')
                        print(f'Invalid response saved: {failure}', flush=True)
                    raise
            else:
                generation, prompt_tokens, latency = 'safe', 0, 0.0
                refusal_pred, violated = False, []
            refusal_gt = parse_bool(row['refusal_gt'])
            batch_output.append({
                **row, 'generation': generation, 'guard_label': 'unsafe' if refusal_pred else 'safe',
                'violated_categories': violated, 'offered_categories': offered,
                'prompt_file': str(prompt_path) if prompt_path else None,
                'prompt_tokens': prompt_tokens, 'inference_seconds': round(latency, 3),
                'refusal_pred': refusal_pred, 'refusal_gt': refusal_gt,
                'correct': refusal_pred == refusal_gt,
            })
            print(f'{len(completed) + len(batch_output)}/{len(data)} | {row["id"]} | '
                  f'{prompt_tokens:,} tokens | {latency:.1f}s | {generation.strip()!r}', flush=True)
        append_jsonl_file(str(output_path), batch_output)
        completed.update({r['id']: r for r in batch_output})
        ordered = [completed[r['id']] for r in data if r['id'] in completed]
        save_evaluation(score_outputs(ordered), str(eval_path))
        print(f'Saved {len(ordered)} / {len(data)} rows.', flush=True)

    final = [completed[r['id']] for r in data if r['id'] in completed]
    metrics = score_outputs(final)
    print_evaluation(metrics)
    save_evaluation(metrics, str(eval_path))
    print(f'Outputs: {output_path}\nEvaluation: {eval_path}\nPrompts: {prompt_dir if args.save_prompts else "disabled"}')


if __name__ == '__main__':
    main()
