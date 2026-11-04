
"""
2026-11-04

Run:
    python ./src/eval/evaluate.py teks_filter

Input:
    data/generated/teks_filter_results/<model>/<query_type>/outputs.jsonl

Output:
    data/evaluation/teks_filter/results.tsv

Only evaluates expectations present in data/splits/test.tsv.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


GENERATED_DIR = Path("data/generated")
EVAL_DIR = Path("data/evaluation")
TEST_PATH = Path("data/splits/test.tsv")

REQUIRED_COLUMNS = {
    "expectation",
    "refusal_gt",
    "refusal_pred",
}

METRIC_COLUMNS = [
    "acc", "prec", "rec",
    "refusal_f1", "allow_f1", "macro_f1",
]


def parse_bool(series):
    values = series.astype(str).str.strip().str.lower()

    if not values.isin(["true", "false"]).all():
        invalid = values[~values.isin(["true", "false"])].unique()
        raise ValueError(
            f"Invalid boolean values in {series.name}: {invalid}"
        )

    return values.eq("true")


def evaluate_group(df):
    y_true = df["refusal_gt"]
    y_pred = df["refusal_pred"]

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "acc": (tp + tn) / len(df),
        "prec": precision,
        "rec": recall,
        "refusal_f1": f1_score(
            y_true, y_pred, pos_label=True, zero_division=0
        ),
        "allow_f1": f1_score(
            y_true, y_pred, pos_label=False, zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate model evaluation results by method."
    )
    parser.add_argument(
        "method",
        help="Evaluation method (e.g., teks_filter)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = GENERATED_DIR / f"{args.method}_results"
    output_dir = EVAL_DIR / args.method
    output_path = output_dir / "results.tsv"

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # ---------------------------------------------------------
    # Load test expectations
    # ---------------------------------------------------------

    test_df = pd.read_csv(TEST_PATH, sep="\t", dtype=str)

    if "expectation" not in test_df.columns:
        raise ValueError(
            f"Missing expectation column in {TEST_PATH}"
        )

    test_expectations = set(test_df["expectation"].dropna())

    if not test_expectations:
        raise ValueError("No expectations found in test.tsv")

    print(f"Test expectations: {len(test_expectations):,}")

    # ---------------------------------------------------------
    # Discover evaluation files
    # ---------------------------------------------------------

    files = sorted(input_dir.rglob("outputs.jsonl"))

    if not files:
        raise FileNotFoundError(
            f"No outputs.jsonl files found in {input_dir}"
        )

    dataframes = []

    for file in files:
        # Expected path:
        # <method>_results/<model>/<query_type>/outputs.jsonl

        model = file.parent.parent.name
        query_type = file.parent.name

        df = pd.read_json(file, lines=True, dtype=False)

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{file}: Missing columns {sorted(missing)}"
            )

        # -----------------------------------------------------
        # Filter to test expectations
        # -----------------------------------------------------

        original_count = len(df)

        df = df[
            df["expectation"].isin(test_expectations)
        ].copy()

        print(
            f"{model} | {query_type}: "
            f"{len(df):,} / {original_count:,} test rows"
        )

        if df.empty:
            continue

        df["model"] = model
        df["query_type"] = query_type

        dataframes.append(df)

    if not dataframes:
        raise ValueError(
            "No evaluation rows match the test expectations."
        )

    df = pd.concat(dataframes, ignore_index=True)

    assert set(df["expectation"]).issubset(test_expectations)

    # ---------------------------------------------------------
    # Normalize labels
    # ---------------------------------------------------------

    df["refusal_gt"] = parse_bool(df["refusal_gt"])
    df["refusal_pred"] = parse_bool(df["refusal_pred"])

    # ---------------------------------------------------------
    # Check duplicates
    # ---------------------------------------------------------

    if "id" in df.columns:
        duplicates = df.duplicated(
            subset=["model", "query_type", "id"],
            keep=False
        )

        if duplicates.any():
            raise ValueError(
                f"Found {duplicates.sum():,} duplicate prediction rows."
            )

    # ---------------------------------------------------------
    # Calculate metrics by model and query type
    # ---------------------------------------------------------

    results = []

    for (model, query_type), group in df.groupby(
        ["model", "query_type"], sort=False
    ):
        results.append({
            "Model": model,
            "query_type": query_type,
            **evaluate_group(group),
        })

    summary = pd.DataFrame(results)

    summary = summary[
        [
            "Model", "query_type",
            "TP", "TN", "FP", "FN",
            *METRIC_COLUMNS,
        ]
    ]

    summary[METRIC_COLUMNS] = summary[METRIC_COLUMNS].round(3)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, sep="\t", index=False)

    print(f"\nSaved: {output_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
