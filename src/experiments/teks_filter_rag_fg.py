"""
2026-11-05
Fine-grained HECAT/TEKS filter with cosine-similarity retrieval.
Permissions: 'x' blocked, 'o' allowed, using fg_expectation_idx.json.
Age-inappropriate expectations are blocked regardless of parental permission.

Run:
  python ./src/experiments/teks_filter_rag_fg.py \
    --input_data_path ./data/splits/fg_split/test.tsv \
    --query_row situation --retrieval_corpus split --backend vllm \
    --model google/gemma-4-12B-it --port 8002 --n_r_contexts 20
"""
from pathlib import Path
import sys
sys.path.append('./src')

import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from utils.file_io import (
    load_tsv_file, load_single_yaml_prompt, append_jsonl_file,
    save_args_as_config_json,
)
from utils.prompting import generate
from utils.evaluation import (
    evaluate_generation, evaluate_dataset, print_evaluation, save_evaluation,
)

TEKS_FILE = 'data/cleaned/hecat_standards.tsv'
IDX_PATH = 'data/generated/fg_expectation_idx.json'
PROMPT_PATH = 'data/prompts/response_generation/teks_guardrail.yaml'
OUTPUT_DIR = 'data/generated/teks_filter_rag_fg_results'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_data_path', default='./data/splits/fg_split/test.tsv')
    p.add_argument('--query_row', choices=['situation', 'eduq'], required=True)
    p.add_argument('--retrieval_corpus', choices=['full', 'split'], default='full')
    p.add_argument('--backend', choices=['openai', 'vllm'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--port', type=int, default=8002)
    p.add_argument('--n_rows', type=int, default=-1)
    p.add_argument('--save_every', type=int, default=25)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--n_r_contexts', type=int, default=20)
    p.add_argument('--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2')
    return p.parse_args()


def normalize(value):
    return ' '.join(str(value or '').split())


def load_index():
    with open(IDX_PATH, encoding='utf-8') as f:
        idx = json.load(f)
    if not isinstance(idx, dict) or sorted(idx.values()) != list(range(len(idx))):
        raise ValueError(f'Invalid expectation index: {IDX_PATH}')
    if any(normalize(k) != k for k in idx):
        raise ValueError('Index contains non-normalized expectations')
    return idx


def load_expectation_catalog(data, retrieval_corpus, idx):
    if retrieval_corpus == 'full':
        teks_df = pd.DataFrame(load_tsv_file(TEKS_FILE))
    else:
        teks_df = pd.DataFrame(data)
    required = {'expectation', 'grade_code'}
    if not required.issubset(teks_df.columns):
        raise ValueError(f'Catalog missing: {sorted(required - set(teks_df.columns))}')
    teks_df['expectation'] = teks_df['expectation'].map(normalize)
    teks_df = teks_df[teks_df['expectation'].ne('')].copy()
    teks_df['grade_code'] = pd.to_numeric(teks_df['grade_code'], errors='raise')
    teks_df = teks_df.drop_duplicates(subset=['expectation']).reset_index(drop=True)
    if teks_df.empty:
        raise ValueError('Empty retrieval catalog')
    # Expectations not present in the saved index are blocked by default.
    teks_df['fg_idx'] = teks_df['expectation'].map(idx)
    return teks_df


class ExpectationRetriever:
    def __init__(self, teks_df, model_name):
        self.teks_df = teks_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)
        self.embeddings = np.asarray(self.model.encode(
            self.teks_df['expectation'].tolist(),
            normalize_embeddings=True, show_progress_bar=True,
        ))

    def retrieve_batch(self, queries, top_k):
        query_embeddings = np.asarray(self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False,
        ))
        scores = query_embeddings @ self.embeddings.T
        k = min(top_k, len(self.teks_df))
        result = []
        for query_scores in scores:
            indices = np.argsort(-query_scores, kind='stable')[:k]
            retrieved = self.teks_df.iloc[indices].copy()
            retrieved['similarity'] = query_scores[indices]
            result.append(retrieved)
        return result


def get_expectation_permissions(row, retrieved_df):
    """Read each retrieved expectation's permission directly from config."""
    config = row['config']
    age = int(row['pretend_age'])
    allowed, disallowed = [], []
    for exp in retrieved_df.itertuples(index=False):
        index = exp.fg_idx
        is_allowed = (
            pd.notna(index)
            and config[int(index)] == 'o'
            and int(exp.grade_code) <= age
        )
        (allowed if is_allowed else disallowed).append(exp.expectation)
    return allowed, disallowed


def build_prompt(row, retrieved_df, prompt_template, query_row):
    allowed, disallowed = get_expectation_permissions(row, retrieved_df)
    return [
        {'role': 'system', 'content': prompt_template['system_prompt'].format(
            allowed_teks='\n'.join(f'- {x}' for x in allowed) or 'None',
            disallowed_teks='\n'.join(f'- {x}' for x in disallowed) or 'None',
        )},
        {'role': 'user', 'content': prompt_template['user_prompt'].format(
            question=row[query_row],
        )},
    ]


def get_retrieval_metadata(row, retrieved_df):
    allowed, _ = get_expectation_permissions(row, retrieved_df)
    allowed_set = set(allowed)
    target_idx = int(row['expectation_idx'])
    retrieval = []
    for exp in retrieved_df.itertuples(index=False):
        exp_idx = None if pd.isna(exp.fg_idx) else int(exp.fg_idx)
        retrieval.append({
            'code': str(getattr(exp, 'code', '')),
            'expectation': exp.expectation,
            'module_code': str(getattr(exp, 'module_code', '')),
            'grade_code': int(exp.grade_code),
            'similarity': float(exp.similarity),
            'expectation_idx': exp_idx,
            'is_target': exp_idx == target_idx,
            'permission': 'allow' if exp.expectation in allowed_set else 'block',
        })
    return retrieval, any(item['is_target'] for item in retrieval)


def load_existing_outputs(path):
    if not path.exists():
        return {}
    outputs = {}
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON: {path}:{line_number}') from e
            if 'id' not in row:
                raise ValueError(f'Missing ID: {path}:{line_number}')
            outputs[row['id']] = row
    return outputs


def validate_input(data, args, idx):
    required = {'id', args.query_row, 'expectation', 'expectation_idx',
                'config', 'grade_code', 'pretend_age', 'refusal_gt'}
    ids = []
    for i, row in enumerate(data, 1):
        missing = required - row.keys()
        if missing:
            raise ValueError(f'Missing fields in row {i}: {missing}')
        if not normalize(row[args.query_row]):
            raise ValueError(f'Empty query in row {i}')
        target = normalize(row['expectation'])
        if target not in idx or int(row['expectation_idx']) != idx[target]:
            raise ValueError(f'Incorrect target expectation index in row {i}')
        config = row['config']
        if len(config) != len(idx) or set(config) - {'x', 'o'}:
            raise ValueError(f'Invalid config in row {i}')
        age = int(row['pretend_age'])
        grade = int(row['grade_code'])
        expected_refusal = config[idx[target]] == 'x' or age < grade
        if (str(row['refusal_gt']).lower() == 'true') != expected_refusal:
            raise ValueError(f'Incorrect refusal_gt in row {i}')
        ids.append(row['id'])
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate dataset IDs')


def main():
    args = parse_args()
    if args.save_every <= 0 or args.n_r_contexts <= 0:
        raise ValueError('--save_every and --n_r_contexts must be positive')
    idx = load_index()
    data = load_tsv_file(args.input_data_path)
    prompt_template = load_single_yaml_prompt(PROMPT_PATH)
    # Use the whole split as the retrieval corpus, even with --n_rows.
    teks_df = load_expectation_catalog(data, args.retrieval_corpus, idx)
    if args.n_rows > 0:
        data = data[:args.n_rows]
    if not data:
        print('No rows found.')
        return
    validate_input(data, args, idx)

    safe_model = args.model.replace('/', '--')
    safe_embedding = args.embedding_model.replace('/', '--')
    run_dir = (Path(OUTPUT_DIR) / safe_model / args.query_row / args.retrieval_corpus
               / f'top{args.n_r_contexts}_{safe_embedding}'
               / Path(args.input_data_path).stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path, eval_path = run_dir / 'outputs.jsonl', run_dir / 'eval.json'
    save_args_as_config_json(args, str(output_path))
    existing = load_existing_outputs(output_path)
    selected_ids = {r['id'] for r in data}
    completed = {k: v for k, v in existing.items() if k in selected_ids}
    remaining = [r for r in data if r['id'] not in completed]
    print('=== FINE-GRAINED TEKS FILTER + RAG ===')
    print(f'Model: {args.model} | Backend: {args.backend}')
    print(f'Loaded: {len(data):,} | Already completed: {len(completed):,} | Remaining: {len(remaining):,}')
    print(f'Catalog: {len(teks_df):,} | Global index: {len(idx):,} | Top-K: {args.n_r_contexts}')
    print(f'Retrieval corpus: {args.retrieval_corpus} | Query: {args.query_row}')

    retriever = ExpectationRetriever(teks_df, args.embedding_model) if remaining else None
    example_printed = False
    with tqdm(total=len(data), initial=len(completed), desc='Generating', unit='row') as progress:
        for start in range(0, len(remaining), args.save_every):
            batch = remaining[start:start + args.save_every]
            retrieved = retriever.retrieve_batch([r[args.query_row] for r in batch], args.n_r_contexts)
            messages = [build_prompt(r, t, prompt_template, args.query_row)
                        for r, t in zip(batch, retrieved, strict=True)]
            if not example_printed:
                tqdm.write("First query's top retrieved expectations:")
                for rank, exp in enumerate(retrieved[0].head(5).itertuples(index=False), 1):
                    tqdm.write(f'{rank}. ({exp.similarity:.3f}) {exp.expectation}')
                tqdm.write('Example message: ' + str(messages[0]))
                example_printed = True
            generations = generate(
                messages=messages, model=args.model, backend=args.backend,
                vllm_base_url=f'http://127.0.0.1:{args.port}/v1',
                temperature=args.temperature, max_tokens=args.max_tokens,
                top_p=args.top_p,
            )
            if len(generations) != len(batch):
                raise ValueError('Generation count mismatch')
            batch_output = []
            for row, generation, retrieved_df in zip(batch, generations, retrieved, strict=True):
                evaluation = evaluate_generation(row, generation)
                retrieval, target_retrieved = get_retrieval_metadata(row, retrieved_df)
                batch_output.append({**row, **evaluation,
                                     'target_retrieved': target_retrieved,
                                     'retrieved_expectations': retrieval})
            append_jsonl_file(str(output_path), batch_output)
            for row in batch_output:
                completed[row['id']] = row
            current = [completed[r['id']] for r in data if r['id'] in completed]
            metrics = evaluate_dataset(current)
            save_evaluation(metrics, str(eval_path))
            progress.update(len(batch_output))

    final_output = [completed[r['id']] for r in data if r['id'] in completed]
    if final_output:
        metrics = evaluate_dataset(final_output)
        print_evaluation(metrics)
        save_evaluation(metrics, str(eval_path))
        audited = [r for r in final_output if 'target_retrieved' in r]
        if audited:
            n_retrieved = sum(str(r['target_retrieved']).lower() == 'true' for r in audited)
            print(f'Target recall@{args.n_r_contexts}: {n_retrieved:,}/{len(audited):,} '
                  f'({n_retrieved / len(audited):.2%})')
    print(f'Outputs: {output_path}\nEvaluation: {eval_path}\nCompleted: {len(final_output):,}/{len(data):,}')


if __name__ == '__main__':
    main()
