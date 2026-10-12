"""Classify student queries using Llama Guard 4 and restricted HECAT expectations.

Run from project root:
    python ./src/experiments/llama_guard_filter.py --input_data_path ./data/generated/sample.tsv \
        --model meta-llama/Llama-Guard-4-12B --n_rows 3 --save_every 1

Start vLLM separately, e.g.:
    vllm serve meta-llama/Llama-Guard-4-12B --host 127.0.0.1 --port 8002
"""
import sys; sys.path.append('./src')
import os
import re
import json
import argparse
import pandas as pd

from utils.file_io import load_tsv_file, load_single_yaml_prompt, append_jsonl_file, save_args_as_config_json
from utils.prompting import generate
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
    p.add_argument('--backend', choices=['vllm'], default='vllm')
    p.add_argument('--vllm_base_url', default='http://localhost:8002/v1')
    p.add_argument('--query_column', default='situation')
    p.add_argument('--prompt_path', default=PROMPT_PATH)
    p.add_argument('--n_rows', type=int, default=-1)
    p.add_argument('--save_every', type=int, default=25)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max_tokens', type=int, default=64)
    p.add_argument('--top_p', type=float, default=1.0)
    return p.parse_args()


def build_prompt(row: dict, teks_df: pd.DataFrame, prompt_template: dict, query_column: str) -> tuple[list[dict], list[str]]:
    pretend_age = int(row['pretend_age'])
    blocked = {x.strip() for x in str(row.get('blocked_cats', '') or '').split(',') if x.strip()}
    if blocked - ALL_CATS.keys():
        raise ValueError(f'Unknown blocked categories: {blocked - ALL_CATS.keys()}')

    disallowed = teks_df.loc[
        teks_df['module_code'].isin(blocked) | (teks_df['grade_code'] > pretend_age)
    ]
    parts = []
    for code, group in disallowed.groupby('module_code', sort=True):
        expectations = '\n'.join(f'- {x}' for x in group['expectation'].drop_duplicates())
        parts.append(f'{code}: {ALL_CATS[code]}.\nRestricted expectations:\n{expectations}')
    text = '\n\n'.join(parts) if parts else 'None'
    prompt = prompt_template['prompt'].format(disallowed_teks=text, question=row[query_column])
    return [{'role': 'user', 'content': prompt}], sorted(disallowed['module_code'].unique().tolist())


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


def load_existing_outputs(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    outputs = {}
    with open(path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if 'id' not in row or row['id'] in outputs:
                raise ValueError(f'Missing/duplicate ID in {path}, line {line_num}')
            outputs[row['id']] = row
    return outputs


def score_outputs(rows: list[dict]) -> dict:
    # Existing evaluator parses the exact refusal sentence from 'generation'.
    # Convert classifier decisions just for scoring; preserve raw guard outputs on disk.
    proxy = [{**row, 'generation': REFUSAL if row['refusal_pred'] else 'safe'} for row in rows]
    return evaluate_dataset(proxy)


def main():
    args = parse_args()
    if args.save_every < 1:
        raise ValueError('--save_every must be >= 1')
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

    safe_model = args.model.replace('/', '--')
    dataset_name = os.path.splitext(os.path.basename(args.input_data_path))[0]
    run_dir = os.path.join(OUTPUT_DIR, safe_model, dataset_name, args.query_column)
    os.makedirs(run_dir, exist_ok=True)
    output_path = os.path.join(run_dir, 'outputs.jsonl')
    eval_path = os.path.join(run_dir, 'eval.json')
    existing = load_existing_outputs(output_path)
    selected = set(ids)
    completed = {k: v for k, v in existing.items() if k in selected}
    remaining = [r for r in data if r['id'] not in completed]
    if existing and any('refusal_pred' not in r for r in completed.values()):
        raise ValueError('Existing output is missing classifier results')
    if not existing:
        save_args_as_config_json(args, output_path)
    print(f'Loaded: {len(data)} | Completed: {len(completed)} | Remaining: {len(remaining)}')

    for start in range(0, len(remaining), args.save_every):
        batch = remaining[start:start + args.save_every]
        prepared = [build_prompt(r, teks_df, template, args.query_column) for r in batch]
        generations = ['safe'] * len(batch)
        infer_indices = [i for i, (_, cats) in enumerate(prepared) if cats]
        if infer_indices:
            messages = [prepared[i][0] for i in infer_indices]
            produced = generate(
                messages=messages, model=args.model, backend=args.backend,
                vllm_base_url=args.vllm_base_url, temperature=args.temperature,
                max_tokens=args.max_tokens, top_p=args.top_p
            )
            if len(produced) != len(infer_indices):
                raise ValueError('Generation count mismatch')
            for i, gen in zip(infer_indices, produced):
                generations[i] = gen

        batch_output = []
        for row, generation, (_, offered) in zip(batch, generations, prepared):
            refusal_pred, violated = parse_guard_output(generation, offered)
            refusal_gt = parse_bool(row['refusal_gt'])
            batch_output.append({
                **row, 'generation': generation, 'guard_label': 'unsafe' if refusal_pred else 'safe',
                'violated_categories': violated, 'offered_categories': offered,
                'refusal_pred': refusal_pred, 'refusal_gt': refusal_gt,
                'correct': refusal_pred == refusal_gt,
            })
        # Validate full batch before writing so an unparseable result can be retried.
        append_jsonl_file(output_path, batch_output)
        completed.update({r['id']: r for r in batch_output})
        ordered = [completed[r['id']] for r in data if r['id'] in completed]
        save_evaluation(score_outputs(ordered), eval_path)
        print(f'Saved {len(ordered)} / {len(data)} rows.', flush=True)

    final = [completed[r['id']] for r in data if r['id'] in completed]
    metrics = score_outputs(final)
    print_evaluation(metrics)
    save_evaluation(metrics, eval_path)
    print(f'Outputs: {output_path}\nEvaluation: {eval_path}')


if __name__ == '__main__':
    main()
