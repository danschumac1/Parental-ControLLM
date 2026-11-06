"""
2026-11-05
Fine-grained TEKS/HECAT retrieval + generation.
'x' blocks, 'o' allows, indexed by data/generated/fg_expectation_idx.json.
Age-inappropriate expectations are always blocked.

Run:
  python ./src/experiments/teks_filter_rag_fg.py \
      --input_data_path ./data/splits/fg_split/test.tsv \
      --query_row situation --retrieval_corpus split --backend vllm \
      --model google/gemma-4-12B-it --port 8002 --n_r_contexts 20

Use --reset_run once if this experiment already has outputs from an older dataset.
It archives old outputs before starting over. Otherwise, completed runs resume safely.
"""

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, './src')
from utils.file_io import (load_tsv_file, load_single_yaml_prompt,
                           append_jsonl_file, save_args_as_config_json)
from utils.prompting import generate
from utils.evaluation import (evaluate_generation, evaluate_dataset,
                              print_evaluation, save_evaluation)

TEKS_FILE = Path('data/cleaned/hecat_standards.tsv')
IDX_PATH = Path('data/generated/fg_expectation_idx.json')
PROMPT_PATH = Path('data/prompts/response_generation/teks_guardrail.yaml')
OUTPUT_DIR = Path('data/generated/teks_filter_rag_fg_results')


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
    p.add_argument('--eval_every', type=int, default=100,
                   help='Evaluate after this many NEW rows (also at finish).')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--max_tokens', type=int, default=1024)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--n_r_contexts', type=int, default=20)
    p.add_argument('--embedding_model', default='sentence-transformers/all-MiniLM-L6-v2')
    p.add_argument('--reset_run', action='store_true',
                   help='Archive earlier outputs/config and regenerate this run.')
    return p.parse_args()


def normalize(value):
    return ' '.join(str(value or '').split())


def load_index():
    with IDX_PATH.open(encoding='utf-8') as f:
        idx = json.load(f)
    if not isinstance(idx, dict) or sorted(idx.values()) != list(range(len(idx))):
        raise ValueError(f'Invalid expectation index: {IDX_PATH}')
    if any(normalize(text) != text for text in idx):
        raise ValueError('Expectation index contains non-normalized text')
    return idx


def load_catalog(all_rows, corpus, idx):
    df = pd.DataFrame(load_tsv_file(TEKS_FILE)) if corpus == 'full' else pd.DataFrame(all_rows)
    required = {'expectation', 'grade_code'}
    if not required.issubset(df.columns):
        raise ValueError(f'Retrieval catalog missing {required - set(df.columns)}')
    df['expectation'] = df['expectation'].map(normalize)
    df = df[df['expectation'].ne('')].copy()
    df['grade_code'] = pd.to_numeric(df['grade_code'], errors='raise').astype(int)
    df = df.drop_duplicates('expectation').reset_index(drop=True)
    if df.empty:
        raise ValueError('Empty retrieval catalog')
    df['fg_idx'] = df['expectation'].map(idx)
    return df


class ExpectationRetriever:
    def __init__(self, catalog, embedding_model):
        self.catalog = catalog
        self.model = SentenceTransformer(embedding_model)
        self.embeddings = np.asarray(self.model.encode(
            catalog['expectation'].tolist(), normalize_embeddings=True,
            show_progress_bar=True,
        ))

    def retrieve_batch(self, queries, top_k):
        embeddings = np.asarray(self.model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False,
        ))
        similarities = embeddings @ self.embeddings.T
        result = []
        for scores in similarities:
            indices = np.argsort(-scores, kind='stable')[:min(top_k, len(self.catalog))]
            retrieved = self.catalog.iloc[indices].copy()
            retrieved['similarity'] = scores[indices]
            result.append(retrieved)
        return result


def permission(row, exp):
    """Read policy character; unknown expectations and immature grades are blocked."""
    index = exp.fg_idx
    return (pd.notna(index)
            and row['config'][int(index)] == 'o'
            and int(exp.grade_code) <= int(row['pretend_age']))


def build_prompt(row, retrieved, template, query_row):
    allowed, blocked = [], []
    for exp in retrieved.itertuples(index=False):
        (allowed if permission(row, exp) else blocked).append(exp.expectation)
    return [
        {'role': 'system', 'content': template['system_prompt'].format(
            allowed_teks='\n'.join(f'- {x}' for x in allowed) or 'None',
            disallowed_teks='\n'.join(f'- {x}' for x in blocked) or 'None')},
        {'role': 'user', 'content': template['user_prompt'].format(question=row[query_row])},
    ]


def retrieval_metadata(row, retrieved):
    target_idx = int(row['expectation_idx'])
    result = []
    for exp in retrieved.itertuples(index=False):
        index = None if pd.isna(exp.fg_idx) else int(exp.fg_idx)
        result.append({
            'code': str(getattr(exp, 'code', '')),
            'expectation': exp.expectation,
            'module_code': str(getattr(exp, 'module_code', '')),
            'grade_code': int(exp.grade_code),
            'similarity': float(exp.similarity),
            'expectation_idx': index,
            'is_target': index == target_idx,
            'permission': 'allow' if permission(row, exp) else 'block',
        })
    return result, any(r['is_target'] for r in result)


def validate_input(rows, args, idx):
    required = {'id', args.query_row, 'expectation', 'expectation_idx',
                'config', 'grade_code', 'pretend_age', 'refusal_gt'}
    ids = set()
    for i, row in enumerate(rows, 1):
        if required - row.keys():
            raise ValueError(f'Missing columns in row {i}: {required - row.keys()}')
        if not normalize(row[args.query_row]):
            raise ValueError(f'Empty {args.query_row} in row {i}')
        target = normalize(row['expectation'])
        if target not in idx or int(row['expectation_idx']) != idx[target]:
            raise ValueError(f'Incorrect expectation index in row {i}')
        config = row['config']
        if len(config) != len(idx) or set(config) - {'x', 'o'}:
            raise ValueError(f'Invalid config in row {i}')
        expected = config[idx[target]] == 'x' or int(row['pretend_age']) < int(row['grade_code'])
        if str(row['refusal_gt']).lower() not in {'true', 'false'}:
            raise ValueError(f'Invalid refusal_gt in row {i}')
        if (str(row['refusal_gt']).lower() == 'true') != expected:
            raise ValueError(f'Incorrect refusal_gt in row {i}')
        if row['id'] in ids:
            raise ValueError(f'Duplicate ID: {row["id"]}')
        ids.add(row['id'])


def run_fingerprint(args, rows, catalog, idx, template):
    """Include inputs that would change retrieval, prompts, or generations."""
    payload = {
        'index': idx,
        'catalog': catalog[['expectation', 'grade_code']].to_dict('records'),
        'rows': [{k: row.get(k) for k in ('id', 'expectation', 'expectation_idx',
                 'config', 'pretend_age', 'grade_code', 'refusal_gt', args.query_row)}
                 for row in rows],
        'prompt': template,
        'generation': {k: getattr(args, k) for k in (
            'query_row', 'retrieval_corpus', 'backend', 'model', 'embedding_model',
            'n_r_contexts', 'temperature', 'max_tokens', 'top_p')},
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def load_existing(path):
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'Invalid JSON {path}:{line_number}') from e
            if not row.get('id'):
                raise ValueError(f'Missing ID {path}:{line_number}')
            result[row['id']] = row
    return result


def prepare_run(run_dir, fingerprint, reset_run):
    """Refuse silent reuse across dataset/index/catalog changes."""
    output_path = run_dir / 'outputs.jsonl'
    manifest_path = run_dir / 'run_fingerprint.json'
    if reset_run:
        backup_dir = run_dir / ('backup_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        for path in run_dir.iterdir():
            if path.is_file():
                backup_dir.mkdir(exist_ok=True)
                shutil.move(str(path), str(backup_dir / path.name))
        if backup_dir.exists():
            print(f'Archived previous run files: {backup_dir}')
    if output_path.exists() and not manifest_path.exists():
        raise ValueError('Existing outputs predate fingerprint tracking. '
                         'Use --reset_run once after rebuilding the dataset.')
    if manifest_path.exists():
        with manifest_path.open(encoding='utf-8') as f:
            saved = json.load(f)
        if saved.get('sha256') != fingerprint:
            raise ValueError('Dataset, index, retrieval corpus, prompt, or model settings changed. '
                             'Use --reset_run to archive and regenerate this experiment.')
    else:
        with manifest_path.open('w', encoding='utf-8') as f:
            json.dump({'sha256': fingerprint}, f, indent=2)
    return output_path, run_dir / 'eval.json'


def main():
    args = parse_args()
    if min(args.save_every, args.eval_every, args.n_r_contexts) <= 0:
        raise ValueError('save_every, eval_every and n_r_contexts must be positive')
    idx = load_index()
    all_rows = load_tsv_file(args.input_data_path)
    template = load_single_yaml_prompt(str(PROMPT_PATH))
    catalog = load_catalog(all_rows, args.retrieval_corpus, idx)
    rows = all_rows[:args.n_rows] if args.n_rows > 0 else all_rows
    if not rows:
        print('No rows found.')
        return
    validate_input(rows, args, idx)

    safe_model = args.model.replace('/', '--')
    safe_embedding = args.embedding_model.replace('/', '--')
    run_dir = (OUTPUT_DIR / safe_model / args.query_row / args.retrieval_corpus
               / f'top{args.n_r_contexts}_{safe_embedding}'
               / Path(args.input_data_path).stem)
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = run_fingerprint(args, rows, catalog, idx, template)
    output_path, eval_path = prepare_run(run_dir, fingerprint, args.reset_run)
    save_args_as_config_json(args, str(output_path))

    existing = load_existing(output_path)
    row_ids = {r['id'] for r in rows}
    completed = {k: v for k, v in existing.items() if k in row_ids}
    remaining = [r for r in rows if r['id'] not in completed]
    print('=== FINE-GRAINED TEKS FILTER RAG ===')
    print(f'Model: {args.model} | Backend: {args.backend} | Query: {args.query_row}')
    print(f'Rows: {len(rows):,} | Completed: {len(completed):,} | Remaining: {len(remaining):,}')
    print(f'Corpus: {args.retrieval_corpus} ({len(catalog):,}) | Index: {len(idx):,} | K={args.n_r_contexts}')

    retriever = ExpectationRetriever(catalog, args.embedding_model) if remaining else None
    since_eval = 0
    example_printed = False
    with tqdm(total=len(rows), initial=len(completed), desc='Generating', unit='row') as bar:
        for start in range(0, len(remaining), args.save_every):
            batch = remaining[start:start + args.save_every]
            retrieved = retriever.retrieve_batch([r[args.query_row] for r in batch], args.n_r_contexts)
            messages = [build_prompt(r, t, template, args.query_row)
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
                temperature=args.temperature, max_tokens=args.max_tokens, top_p=args.top_p,
            )
            if len(generations) != len(batch):
                raise ValueError('Generation count mismatch')

            batch_output = []
            for row, generation, retrieved_df in zip(batch, generations, retrieved, strict=True):
                evaluation = evaluate_generation(row, generation)
                retrieval, target_retrieved = retrieval_metadata(row, retrieved_df)
                batch_output.append({**row, **evaluation, 'target_retrieved': target_retrieved,
                                     'retrieved_expectations': retrieval})
            append_jsonl_file(str(output_path), batch_output)
            completed.update({r['id']: r for r in batch_output})
            since_eval += len(batch_output)
            bar.update(len(batch_output))
            if since_eval >= args.eval_every:
                current = [completed[r['id']] for r in rows if r['id'] in completed]
                save_evaluation(evaluate_dataset(current), str(eval_path))
                since_eval = 0

    final = [completed[r['id']] for r in rows if r['id'] in completed]
    if final:
        metrics = evaluate_dataset(final)
        print_evaluation(metrics)
        save_evaluation(metrics, str(eval_path))
        audited = [r for r in final if 'target_retrieved' in r]
        if audited:
            retrieved_count = sum(r['target_retrieved'] is True for r in audited)
            print(f'Target recall@{args.n_r_contexts}: {retrieved_count:,}/{len(audited):,} '
                  f'({retrieved_count / len(audited):.2%})')
    print(f'Outputs: {output_path}\nEvaluation: {eval_path}\nCompleted: {len(final):,}/{len(rows):,}')


if __name__ == '__main__':
    main()
