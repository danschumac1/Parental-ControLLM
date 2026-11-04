'''
2026-11-04

Creates train/dev/test splits such that:
1. Each row contains both a situation and an EduQ.
2. All original content and metadata are retained.
3. Each expectation belongs to exactly one split.
4. Splits are approximately 60/20/20.

Run:
    python ./src/data_management/train_dev_test_split.py
'''

import sys; sys.path.append("./src")
import random
from pathlib import Path

import pandas as pd

from utils.file_io import load_tsv_file


# =============================================================================
# CONFIGURATION
# =============================================================================

FULL_SITUATIONS_PATH = "data/generated/situations.tsv"
FULL_EDUQ_PATH = "data/generated/eduQ.tsv"

OUTPUT_DIR = Path("data/splits")

RANDOM_STATE = 42

P_TRAIN = 60
P_DEV = 20
P_TEST = 20

assert P_TRAIN + P_DEV + P_TEST == 100


# =============================================================================
# HELPERS
# =============================================================================

def split_expectations(expectations):
    """Randomly assign unique expectations to train/dev/test."""

    expectations = list(expectations)

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
    """Check for expectation leakage, duplicate IDs, and lost rows."""

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

    # No IDs may appear in multiple splits.
    ids = {
        name: set(df["id"])
        for name, df in splits.items()
    }

    assert ids["train"].isdisjoint(ids["dev"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["dev"].isdisjoint(ids["test"])

    # No rows or expectations are lost.
    assert sum(len(df) for df in splits.values()) == len(full_df)

    assert set().union(*expectations.values()) == set(full_df["expectation"])

    # Every row must contain both types of requests.
    for name, df in splits.items():
        assert df["situation"].notna().all(), f"Missing situations in {name}"
        assert df["eduq"].notna().all(), f"Missing EduQs in {name}"

    print("All split validations passed.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    situ_df = pd.DataFrame(load_tsv_file(FULL_SITUATIONS_PATH))
    eduq_df = pd.DataFrame(load_tsv_file(FULL_EDUQ_PATH))

    print(f"Situation rows: {len(situ_df):,}")
    print(f"EduQ rows:      {len(eduq_df):,}")

    print(f"Situation expectations: {situ_df['expectation'].nunique():,}")
    print(f"EduQ expectations:      {eduq_df['expectation'].nunique():,}")

    # -------------------------------------------------------------------------
    # Normalize IDs for pairing
    # -------------------------------------------------------------------------
    #
    # Situation ID:
    #   google__gemma-4-12B-it__000290__all_allowed__inappropriate
    #
    # EduQ ID:
    #   eduQ__google__gemma-4-12B-it__000290__all_allowed__inappropriate
    #
    # Removing "eduQ__" gives us a shared pairing key.
    # -------------------------------------------------------------------------

    situ_df["pair_id"] = situ_df["id"]

    assert eduq_df["id"].str.startswith("eduQ__").all(), (
        "Some EduQ IDs do not start with 'eduQ__'."
    )

    eduq_df["pair_id"] = eduq_df["id"].str.removeprefix("eduQ__")

    assert situ_df["pair_id"].is_unique, "Duplicate situation IDs."
    assert eduq_df["pair_id"].is_unique, "Duplicate EduQ pairing IDs."

    situ_ids = set(situ_df["pair_id"])
    eduq_ids = set(eduq_df["pair_id"])

    assert situ_ids == eduq_ids, (
        f"Situation/EduQ pairing IDs do not match.\n"
        f"Only in situations: {len(situ_ids - eduq_ids)}\n"
        f"Only in EduQ: {len(eduq_ids - situ_ids)}"
    )

    # -------------------------------------------------------------------------
    # Merge situations and EduQs
    # -------------------------------------------------------------------------

    eduq_df = eduq_df.rename(columns={
        "id": "eduq_id",
        "situation": "eduq",
    })

    df = situ_df.merge(
        eduq_df,
        on="pair_id",
        how="inner",
        suffixes=("", "_eduq"),
        validate="one_to_one"
    )

    assert len(df) == len(situ_df) == len(eduq_df), (
        "Rows were lost or duplicated during merging."
    )

    # Verify all shared metadata matches before dropping duplicate columns.
    duplicate_cols = [
        col for col in df.columns
        if col.endswith("_eduq") and col != "eduq_id"
    ]

    for col in duplicate_cols:
        original = col.removesuffix("_eduq")

        assert (
            df[original].fillna("").astype(str)
            == df[col].fillna("").astype(str)
        ).all(), f"Metadata mismatch: {original}"

    df = df.drop(columns=duplicate_cols)

    # Temporary pairing key is no longer necessary.
    df = df.drop(columns="pair_id")

    print(f"\nSuccessfully paired {len(df):,} situations with EduQs.")

    # -------------------------------------------------------------------------
    # Split by expectation
    # -------------------------------------------------------------------------

    expectations = sorted(df["expectation"].unique())

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

    for name, split_df in [
        ("train", train_df),
        ("dev", dev_df),
        ("test", test_df),
    ]:
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

    for name, split_df in [
        ("TRAIN", train_df),
        ("DEV", dev_df),
        ("TEST", test_df),
    ]:
        print(
            f"{name:<6} "
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