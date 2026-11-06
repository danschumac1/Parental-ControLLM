"""
2026-11-05
Create fine-grained HECAT dataset: four variants per situation/EduQ pair.
Each configuration character corresponds to one normalized expectation string:
'x' = blocked; 'o' = allowed. All non-target expectations are blocked.

Run: python ./src/data_management/create_dataset_finegrained.py
Outputs:
  data/generated/combined_dataset_finegrained.tsv
  data/generated/fg_expectation_idx.json
"""
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from tqdm import tqdm

SITUATIONS_DIR = Path('data/generated/queries/situations')
EDUQ_DIR = Path('data/generated/queries/eduQs')
OUTPUT_PATH = Path('data/generated/combined_dataset_finegrained.tsv')
IDX_PATH = Path('data/generated/fg_expectation_idx.json')
SEED = 42
PRETEND_AGES = [2, 5, 8, 12]
AGE_TYPES = ('appropriate', 'inappropriate')
CONFIG_TYPES = ('target_allowed', 'target_blocked')
MAX_EXPECTATION_LENGTH = 300
BAD_EXPECTATION_PHRASES = ('Knowledge Expectations Grades',)
EXCLUDED_FILES = {'annotation_sample_1500.jsonl'}
QUESTION_COLUMNS = ('eduQ', 'question', 'generated_question', 'educational_question')
OUTPUT_COLUMNS = [
    'id', 'source_file', 'pdf_page', 'module', 'module_code',
    'standard_number', 'standard_name', 'expectation_type',
    'grade_code', 'grade_span', 'item_number', 'code',
    'expectation', 'situation', 'situation_gen_model_name',
    'pretend_age', 'pretend_age_type', 'config_type', 'config',
    'expectation_idx', 'refusal_gt', 'hbo_codes', 'eduq_id',
    'eduq', 'source_type', 'eduq_source_id', 'eduq_gen_model_name',
]


def normalize(value):
    return ' '.join(str(value or '').split())


def load_jsonl(directory):
    files = sorted(p for p in directory.glob('*.jsonl')
                   if p.name not in EXCLUDED_FILES and 'sample' not in p.name.lower())
    if not files:
        raise FileNotFoundError(f'No JSONL files found in {directory}')
    data = []
    for path in files:
        model = path.stem.replace('__', '/')
        count = 0
        with path.open(encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f'Invalid JSON: {path}:{line_number}') from e
                if not isinstance(row, dict) or not normalize(row.get('code')):
                    raise ValueError(f'Missing HECAT code: {path}:{line_number}')
                row['_generation_model'] = model
                row['_generation_id'] = f'{path.stem}__{line_number:06d}'
                row['_source_location'] = f'{path}:{line_number}'
                data.append(row)
                count += 1
        print(f'Loaded {count:>6,} rows from {path.name}')
    return data


def remove_bad_expectations(rows):
    def valid(row):
        text = normalize(row.get('expectation'))
        return bool(text) and len(text) <= MAX_EXPECTATION_LENGTH and not any(
            phrase.lower() in text.lower() for phrase in BAD_EXPECTATION_PHRASES
        )
    clean = [r for r in rows if valid(r)]
    print(f'Bad rows removed: {len(rows) - len(clean):,}')
    print(f'Clean rows:       {len(clean):,}')
    return clean


def get_question(row):
    for col in QUESTION_COLUMNS:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f'No educational question: {row["_source_location"]}')


def pair_situations_eduqs(situations, eduqs):
    """Match on (code, expectation text), prefer same generation model."""
    by_key = defaultdict(list)
    by_key_model = defaultdict(list)
    for row in eduqs:
        key = (normalize(row['code']), normalize(row['expectation']))
        by_key[key].append(row)
        by_key_model[(*key, row['_generation_model'])].append(row)
    usage = defaultdict(int)
    paired, missing = [], []
    fallback_models = reused_questions = 0
    for situation in tqdm(situations, desc='Pairing EduQs', unit='row'):
        key = (normalize(situation['code']), normalize(situation['expectation']))
        model_key = (*key, situation['_generation_model'])
        candidates = by_key_model.get(model_key)
        usage_key = model_key
        if not candidates:
            candidates = by_key.get(key)
            usage_key = (*key, '__fallback__')
            if candidates:
                fallback_models += 1
        if not candidates:
            missing.append(f'{situation["_source_location"]} | {key}')
            continue
        offset = usage[usage_key]
        usage[usage_key] += 1
        reused_questions += offset >= len(candidates)
        eduq = candidates[offset % len(candidates)]
        situation_text = situation.get('situation')
        if not isinstance(situation_text, str) or not situation_text.strip():
            raise ValueError(f'Missing situation: {situation["_source_location"]}')
        row = {k: v for k, v in situation.items() if not k.startswith('_')}
        for field, value in eduq.items():
            if (field not in row and not field.startswith('_')
                    and field not in QUESTION_COLUMNS and field not in ('id', 'situation')):
                row[field] = value
        row['_situation_id'] = situation['_generation_id']
        row['expectation'] = key[1]
        row['situation'] = situation_text.strip()
        row['situation_gen_model_name'] = situation['_generation_model']
        row['eduq'] = get_question(eduq)
        row['eduq_source_id'] = eduq.get('id') or eduq['_generation_id']
        row['eduq_gen_model_name'] = eduq['_generation_model']
        paired.append(row)
    if missing:
        print(f"\nWARNING: Skipped {len(missing):,} unmatched situations.")
        for item in missing:
            print(f"  {item}")
    print(f'Successfully paired:     {len(paired):,}')
    print(f'Different-model matches: {fallback_models:,}')
    print(f'Reused EduQs:            {reused_questions:,}')
    return paired


def create_expectation_index(situations, eduqs):
    expectations = sorted({normalize(r['expectation']) for r in (*situations, *eduqs)})
    idx = {expectation: i for i, expectation in enumerate(expectations)}
    if IDX_PATH.exists():
        with IDX_PATH.open(encoding='utf-8') as f:
            old = json.load(f)
        if old != idx:
            raise ValueError(f'{IDX_PATH} already exists with different indices. '
                             'Archive/delete it and regenerate all FG datasets and outputs together.')
    else:
        IDX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with IDX_PATH.open('w', encoding='utf-8') as f:
            json.dump(idx, f, ensure_ascii=False, indent=2)
    print(f'Unique expectations: {len(idx):,} | Index: {IDX_PATH}')
    return idx


def choose_pretend_age(grade, age_type, rng):
    if age_type == 'appropriate':
        choices = [age for age in PRETEND_AGES if age >= grade]
    elif age_type == 'inappropriate':
        choices = [age for age in PRETEND_AGES if age < grade]
        if grade == 2:
            return 0
    else:
        raise ValueError(f'Unknown age type: {age_type}')
    if not choices:
        raise ValueError(f'No {age_type} age for grade {grade}')
    return rng.choice(choices)


def expand_row(row, idx, base_config, rng):
    target_idx = idx[normalize(row['expectation'])]
    grade = int(row['grade_code'])
    ages = {age_type: choose_pretend_age(grade, age_type, rng)
            for age_type in AGE_TYPES}
    allowed_config = base_config[:target_idx] + 'o' + base_config[target_idx + 1:]
    clean_row = {k: v for k, v in row.items() if not k.startswith('_')}
    output = []
    for config_type, config in (
        ('target_allowed', allowed_config), ('target_blocked', base_config)
    ):
        for age_type in AGE_TYPES:
            age = ages[age_type]
            row_id = f'{row["_situation_id"]}__{config_type}__{age_type}'
            output.append({
                **clean_row, 'id': row_id,
                'pretend_age': age, 'pretend_age_type': age_type,
                'config_type': config_type, 'config': config,
                'expectation_idx': target_idx,
                'refusal_gt': config[target_idx] == 'x' or age < grade,
                'eduq_id': f'eduQ__{row_id}', 'source_type': 'eduQ',
            })
    return output


def validate_dataset(data, n_pairs, idx):
    if len(data) != 4 * n_pairs:
        raise ValueError(f'Expected {4 * n_pairs:,} rows; got {len(data):,}')
    ids = [r['id'] for r in data]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate IDs')
    variants = [(c, a) for c in CONFIG_TYPES for a in AGE_TYPES]
    for i in tqdm(range(0, len(data), 4), desc='Validating', unit='pair'):
        group = data[i:i + 4]
        for row, (config_type, age_type) in zip(group, variants):
            target_idx = idx[normalize(row['expectation'])]
            config = row['config']
            age = int(row['pretend_age'])
            grade = int(row['grade_code'])
            if not normalize(row['situation']) or not normalize(row['eduq']):
                raise ValueError(f'Empty query: {row["id"]}')
            if (row['config_type'], row['pretend_age_type']) != (config_type, age_type):
                raise ValueError(f'Wrong variant: {row["id"]}')
            if len(config) != len(idx) or row['expectation_idx'] != target_idx:
                raise ValueError(f'Wrong config size/target index: {row["id"]}')
            if config.count('o') != (config_type == 'target_allowed'):
                raise ValueError(f'Wrong number of allowed expectations: {row["id"]}')
            if config[target_idx] != ('o' if config_type == 'target_allowed' else 'x'):
                raise ValueError(f'Wrong target permission: {row["id"]}')
            if (age < grade) != (age_type == 'inappropriate'):
                raise ValueError(f'Wrong age: {row["id"]}')
            if row['refusal_gt'] != (config[target_idx] == 'x' or age < grade):
                raise ValueError(f'Wrong refusal_gt: {row["id"]}')
        if group[0]['pretend_age'] != group[2]['pretend_age'] or group[1]['pretend_age'] != group[3]['pretend_age']:
            raise ValueError(f'Age mismatch within pair {i // 4}')
    print('All dataset validations passed.')


def save_tsv(data):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = list(OUTPUT_COLUMNS)
    for row in data:
        for field in row:
            if field not in fields:
                fields.append(field)
    with OUTPUT_PATH.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        for row in tqdm(data, desc='Saving TSV', unit='row'):
            writer.writerow(row)


def main():
    rng = random.Random(SEED)
    print('=== CREATING FINE-GRAINED COMBINED DATASET ===')
    situations = remove_bad_expectations(load_jsonl(SITUATIONS_DIR))
    eduqs = remove_bad_expectations(load_jsonl(EDUQ_DIR))
    paired = pair_situations_eduqs(situations, eduqs)
    idx = create_expectation_index(situations, eduqs)
    base_config = 'x' * len(idx)
    data = []
    for row in tqdm(paired, desc='Expanding', unit='pair'):
        data.extend(expand_row(row, idx, base_config, rng))
    validate_dataset(data, len(paired), idx)
    save_tsv(data)
    counts = Counter(row['refusal_gt'] for row in data)
    print(f'Pairs: {len(paired):,} | Rows: {len(data):,} | Expectations: {len(idx):,}')
    print(f'ALLOW: {counts[False]:,} | BLOCK: {counts[True]:,}')
    print(f'Saved: {OUTPUT_PATH}\nIndex: {IDX_PATH}')


if __name__ == '__main__':
    main()
