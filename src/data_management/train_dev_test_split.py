
"""
2026-11-04

Create train/dev/test splits from the combined dataset.

1. Each row contains both a situation and an EduQ.
2. All original content and metadata are retained.
3. Each expectation belongs to exactly one split.
4. Splits are approximately 60/20/20 by expectation count.

Run:
    python ./src/data_management/train_dev_test_split.py

Input:
    data/generated/combined_dataset.tsv

Outputs:
    data/splits/train.tsv
    data/splits/dev.tsv
    data/splits/test.tsv
"""

import random
from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATH = Path("data/generated/combined_dataset.tsv")
OUTPUT_DIR = Path("data/splits")

RANDOM_STATE = 42

P_TRAIN = 80
P_DEV = 10
P_TEST = 10

assert P_TRAIN + P_DEV + P_TEST == 100


# =============================================================================
# HELPERS
# =============================================================================

def split_expectations(expectations):
    """Randomly assign unique expectations to train/dev/test."""

    expectations = sorted(expectations)

    rng = random.Random(RANDOM_STATE)
    rng.shuffle(expectations)

    n = len(expectations)

    n_train = round(n * P_TRAIN / 100)
    n_dev = round(n * P_DEV / 100)

    train = set(expectations[:n_train])
    dev = set(expectations[n_train:n_train + n_dev])
    test = set(expectations[n_train + n_dev:])

    return train, dev, test


def validate_splits(train_df, dev_df, test_df, full_df):
    """Check for leakage, duplicate IDs, missing data, and lost rows."""

    splits = {
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }

    # No expectation may appear in multiple splits.
    expectations = {
        name: set(df["expectation"])
        for name, df in splits.items()
    }

    assert expectations["train"].isdisjoint(expectations["dev"])
    assert expectations["train"].isdisjoint(expectations["test"])
    assert expectations["dev"].isdisjoint(expectations["test"])

    # No ID may appear in multiple splits.
    ids = {
        name: set(df["id"])
        for name, df in splits.items()
    }

    assert ids["train"].isdisjoint(ids["dev"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["dev"].isdisjoint(ids["test"])

    # No rows or expectations are lost.
    assert sum(len(df) for df in splits.values()) == len(full_df)

    assert (
        set().union(*expectations.values())
        == set(full_df["expectation"])
    )

    # Every row must contain both requests.
    for name, df in splits.items():
        for col in ("situation", "eduq"):
            assert df[col].notna().all(), (
                f"Missing {col} values in {name}"
            )
            assert df[col].astype(str).str.strip().ne("").all(), (
                f"Empty {col} values in {name}"
            )

    print("All split validations passed.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load combined dataset
    # -------------------------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    df = pd.read_csv(INPUT_PATH, sep="\t", dtype=str)

    required = {"id", "expectation", "situation", "eduq"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Combined dataset is empty.")

    if df["id"].isna().any() or not df["id"].is_unique:
        raise ValueError("Missing or duplicate IDs in combined dataset.")

    if (
        df["expectation"].isna().any()
        or df["expectation"].str.strip().eq("").any()
    ):
        raise ValueError("Missing or empty expectations.")

    print(f"Total rows:         {len(df):,}")
    print(f"Total expectations: {df['expectation'].nunique():,}")

    # -------------------------------------------------------------------------
    # Split by expectation
    # -------------------------------------------------------------------------

    expectations = df["expectation"].unique()

    train_exp, dev_exp, test_exp = split_expectations(expectations)

    train_df = df[df["expectation"].isin(train_exp)].copy()
    dev_df = df[df["expectation"].isin(dev_exp)].copy()
    test_df = df[df["expectation"].isin(test_exp)].copy()

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_splits(train_df, dev_df, test_df, df)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_df,
        "dev": dev_df,
        "test": test_df,
    }

    for name, split_df in splits.items():
        split_df.to_csv(
            OUTPUT_DIR / f"{name}.tsv",
            sep="\t",
            index=False
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\nSPLIT SUMMARY")
    print("-" * 65)

    for name, split_df in splits.items():
        print(
            f"{name.upper():<6} "
            f"rows={len(split_df):>6,} | "
            f"expectations={split_df['expectation'].nunique():>5,} | "
            f"rows%={len(split_df) / len(df):>6.2%}"
        )

    print("-" * 65)
    print(f"Total rows:         {len(df):,}")
    print(f"Total expectations: {df['expectation'].nunique():,}")
    print(f"Random seed:        {RANDOM_STATE}")
    print(f"Output directory:   {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
