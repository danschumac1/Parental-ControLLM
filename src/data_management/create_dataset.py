"""Build one combined dataset from all generated situation and EduQ JSONL files.

Run:
    python ./src/data_management/create_dataset.py

Inputs (all three models, excluding annotation samples):
    data/generated/situations/*.jsonl
    data/generated/queries/eduQs/*.jsonl

Output:
    data/generated/combined_dataset.tsv

Pairs by HECAT expectation code; prefers the same generating model and rotates
through questions if an expectation has multiple situations. All original
situation metadata is retained. DOES NOT invent missing experimental conditions
or refusal ground truth: those require separate generation/labeling logic.
"""

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

SITUATIONS_DIR = Path("data/generated/queries/situations")
EDUQ_DIR = Path("data/generated/queries/eduQs")
OUTPUT_PATH = Path('data/generated/combined_dataset.tsv')
EXCLUDED = {'annotation_sample_1500.jsonl'}
QUESTION_COLUMNS = ('eduQ', 'question', 'generated_question', 'educational_question')

OUTPUT_COLUMNS = (
    'id', 'source_file', 'pdf_page', 'module', 'module_code',
    'standard_number', 'standard_name', 'expectation_type', 'grade_code',
    'grade_span', 'item_number', 'code', 'expectation', 'situation',
    'situation_gen_model_name', 'pretend_age', 'pretend_age_type',
    'config_type', 'config', 'blocked_cats', 'refusal_gt', 'hbo_codes',
    'eduq_id', 'eduq', 'source_type', 'eduq_source_id',
    'eduq_gen_model_name',
)


def norm(value):
    return ' '.join(str(value or '').split())


def model_name(path):
    stem = path.stem
    for prefix in ('google_', 'microsoft_'):
        if stem.startswith(prefix):
            return prefix[:-1] + '/' + stem[len(prefix):]
    return stem.replace('__', '/')


def text_column(row, columns, location):
    for col in columns:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f'Missing text in {columns} at {location}')


def load(directory):
    files = [p for p in sorted(directory.glob('*.jsonl')) if p.name not in EXCLUDED]
    if not files:
        raise FileNotFoundError(f'No JSONL files found under {directory}')
    rows = []
    for path in files:
        count = 0
        with path.open(encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'Bad JSON at {path}:{line_no}') from exc
                if not isinstance(row, dict) or not norm(row.get('code')):
                    raise ValueError(f'Missing HECAT code at {path}:{line_no}')
                rows.append((row, model_name(path), f'{path}:{line_no}'))
                count += 1
        print(f'  {path}: {count:,} rows')
    return rows


def create_dataset(situations, questions):
    by_code = defaultdict(list)
    by_code_model = defaultdict(list)
    for item in questions:
        row, model, _ = item
        code = norm(row['code'])
        by_code[code].append(item)
        by_code_model[(code, model)].append(item)

    used = defaultdict(int)
    output = []
    missing = []
    fallback = 0
    reused = 0
    expectation_mismatch = 0

    for index, (s, s_model, s_location) in enumerate(situations, 1):
        code = norm(s['code'])
        same_model = by_code_model.get((code, s_model), [])
        candidates = same_model if same_model else by_code.get(code, [])
        if not candidates:
            missing.append(s_location + ' (' + code + ')')
            continue
        if not same_model:
            fallback += 1
        bucket = (code, s_model if same_model else '__fallback__')
        offset = used[bucket]
        used[bucket] += 1
        if offset >= len(candidates):
            reused += 1
        q, q_model, q_location = candidates[offset % len(candidates)]

        if norm(s.get('expectation')) != norm(q.get('expectation')):
            expectation_mismatch += 1

        # Preserve all situation metadata. Fill in expectation metadata from
        # the EduQ source only when the situation does not contain that field.
        row = dict(s)
        for field, value in q.items():
            if field not in row and field not in (*QUESTION_COLUMNS, 'id', 'situation'):
                row[field] = value

        original_id = norm(s.get('id'))
        row['id'] = original_id or f'{s_model.replace("/", "__")}__{index:08d}'
        row['situation'] = text_column(s, ('situation',), s_location)
        row['eduq'] = text_column(q, QUESTION_COLUMNS, q_location)
        row['situation_gen_model_name'] = norm(s.get('situation_gen_model_name')) or s_model
        row['eduq_gen_model_name'] = norm(q.get('eduq_gen_model_name')) or q_model
        row['eduq_id'] = 'eduQ__' + row['id']
        row['eduq_source_id'] = norm(q.get('id')) or f'{q_model.replace("/", "__")}__{q_location.rsplit(":", 1)[-1]}'
        row['source_type'] = 'eduQ'
        output.append(row)

    if missing:
        raise ValueError(f'{len(missing):,} situations without an EduQ match:\n' + '\n'.join(missing[:10]))
    if not output:
        raise ValueError('No paired rows produced.')

    df = pd.DataFrame(output)
    if not df['id'].is_unique:
        raise ValueError('Duplicate situation IDs. Cannot safely produce paired data.')
    if len(df) != len(situations):
        raise ValueError('Some situations were lost.')
    if df[['situation', 'eduq']].isna().any().any():
        raise ValueError('Missing question text.')
    for col in ('situation', 'eduq'):
        if df[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'Blank {col} text.')

    # These cannot be derived merely by joining the JSONL files.
    for col in ('pretend_age', 'pretend_age_type', 'config_type', 'config', 'blocked_cats', 'refusal_gt'):
        if col not in df.columns or df[col].isna().all():
            print(f'WARNING: {col} not supplied by input JSONL; no values invented.')

    ordered = [c for c in OUTPUT_COLUMNS if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    print(f'Paired rows: {len(df):,} | expectations: {df["code"].nunique():,}')
    print(f'Model fallbacks: {fallback:,} | reused questions: {reused:,}')
    print(f'Expectation text mismatches: {expectation_mismatch:,}')
    return df


def main():
    print('Situation inputs:')
    situations = load(SITUATIONS_DIR)
    print('\nEduQ inputs:')
    questions = load(EDUQ_DIR)
    df = create_dataset(situations, questions)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, sep='\t', index=False)
    print(f'\nSaved: {OUTPUT_PATH} ({len(df):,} rows)')


if __name__ == '__main__':
    main()
