"""
2026-11-05
Split fine-grained dataset by normalized expectation text (80/10/10).
All four variants remain together. Verify config against shared index.

Run: python ./src/data_management/train_dev_test_split_finegrained.py
Input:  data/generated/combined_dataset_finegrained.tsv
Index:  data/generated/fg_expectation_idx.json
Output: data/splits/fg_split/{train,dev,test}.tsv
"""
import json
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

INPUT_PATH = Path('data/generated/combined_dataset_finegrained.tsv')
IDX_PATH = Path('data/generated/fg_expectation_idx.json')
OUTPUT_DIR = Path('data/splits/fg_split')
RANDOM_STATE = 42
P_TRAIN, P_DEV, P_TEST = 80, 10, 10
assert P_TRAIN + P_DEV + P_TEST == 100
VARIANTS = {
    ('target_allowed', 'appropriate'),
    ('target_allowed', 'inappropriate'),
    ('target_blocked', 'appropriate'),
    ('target_blocked', 'inappropriate'),
}


def normalize(value):
    return ' '.join(str(value or '').split())


def split_expectations(expectations):
    expectations = sorted(expectations)
    random.Random(RANDOM_STATE).shuffle(expectations)
    n_train = round(len(expectations) * P_TRAIN / 100)
    n_dev = round(len(expectations) * P_DEV / 100)
    return (
        set(expectations[:n_train]),
        set(expectations[n_train:n_train + n_dev]),
        set(expectations[n_train + n_dev:]),
    )


def validate_splits(splits, full_df, idx):
    exp_sets = {name: set(df['expectation']) for name, df in splits.items()}
    id_sets = {name: set(df['id']) for name, df in splits.items()}
    for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test')):
        if not exp_sets[a].isdisjoint(exp_sets[b]):
            raise ValueError(f'Expectation leakage between {a} and {b}')
        if not id_sets[a].isdisjoint(id_sets[b]):
            raise ValueError(f'ID leakage between {a} and {b}')
    if sum(len(df) for df in splits.values()) != len(full_df):
        raise ValueError('Lost rows while splitting')
    if set().union(*exp_sets.values()) != set(full_df['expectation']):
        raise ValueError('Lost expectations while splitting')

    suffix = r'__(target_allowed|target_blocked)__(appropriate|inappropriate)$'
    for name, df in splits.items():
        if df.empty:
            continue
        for col in ('situation', 'eduq'):
            if df[col].isna().any() or df[col].str.strip().eq('').any():
                raise ValueError(f'Missing {col} in {name}')
        original_ids = df['id'].str.replace(suffix, '', regex=True)
        if (original_ids == df['id']).any():
            raise ValueError(f'Unrecognized variant ID suffix in {name}')
        for original_id, group in tqdm(df.groupby(original_ids, sort=False),
                                       desc=f'Validating {name}', unit='pair'):
            variants = set(zip(group['config_type'], group['pretend_age_type']))
            if len(group) != 4 or variants != VARIANTS:
                raise ValueError(f'Invalid variants for {original_id} in {name}')
            for row in group.itertuples(index=False):
                target_idx = idx[row.expectation]
                if int(row.expectation_idx) != target_idx:
                    raise ValueError(f'Invalid expectation_idx in {row.id}')
                config = row.config
                should_allow = row.config_type == 'target_allowed'
                if (len(config) != len(idx)
                        or config.count('o') != int(should_allow)
                        or config[target_idx] != ('o' if should_allow else 'x')):
                    raise ValueError(f'Invalid config in {row.id}')
                age_bad = int(row.pretend_age) < int(row.grade_code)
                if age_bad != (row.pretend_age_type == 'inappropriate'):
                    raise ValueError(f'Invalid age in {row.id}')
                expected = not should_allow or age_bad
                if (str(row.refusal_gt).lower() == 'true') != expected:
                    raise ValueError(f'Incorrect refusal_gt in {row.id}')
    print('All split validations passed.')


def main():
    print('=== FINE-GRAINED TRAIN/DEV/TEST SPLITS ===')
    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)
    if not IDX_PATH.exists():
        raise FileNotFoundError(IDX_PATH)
    with IDX_PATH.open(encoding='utf-8') as f:
        idx = json.load(f)
    if not isinstance(idx, dict) or sorted(idx.values()) != list(range(len(idx))):
        raise ValueError('Invalid expectation index mapping')
    df = pd.read_csv(INPUT_PATH, sep='\t', dtype=str, keep_default_na=False)
    required = {'id', 'expectation', 'expectation_idx', 'config', 'situation', 'eduq',
                'pretend_age', 'grade_code', 'config_type', 'pretend_age_type', 'refusal_gt'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.empty or df['id'].eq('').any() or not df['id'].is_unique:
        raise ValueError('Empty dataset or invalid IDs')
    df['expectation'] = df['expectation'].map(normalize)
    if df['expectation'].eq('').any():
        raise ValueError('Missing expectation')
    unknown = set(df['expectation']) - set(idx)
    if unknown:
        raise ValueError(f'Expectations missing from index: {list(unknown)[:5]}')

    train_exp, dev_exp, test_exp = split_expectations(df['expectation'].unique())
    splits = {
        'train': df[df['expectation'].isin(train_exp)].copy(),
        'dev': df[df['expectation'].isin(dev_exp)].copy(),
        'test': df[df['expectation'].isin(test_exp)].copy(),
    }
    validate_splits(splits, df, idx)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = OUTPUT_DIR / f'{name}.tsv'
        split_df.to_csv(path, sep='\t', index=False)
        n_allow = split_df['refusal_gt'].str.lower().eq('false').sum()
        print(f'{name.upper():<5} rows={len(split_df):>7,} | '
              f'expectations={split_df["expectation"].nunique():>5,} | '
              f'allow={n_allow:>6,} | block={len(split_df) - n_allow:>6,} | {path}')
    print(f'Total rows: {len(df):,} | Indexed expectations: {len(idx):,}')


if __name__ == '__main__':
    main()
