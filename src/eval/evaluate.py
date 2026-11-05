"""Aggregate parental-control experiment evaluations.

Run:
    python ./src/eval/evaluate.py
    python ./src/eval/evaluate.py --methods teks_filter teks_filter_rag upper_bound_rag

Output:
    data/all_methods_results.tsv

Evaluates all rows in each outputs.jsonl; does not filter against test.tsv.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

GENERATED_DIR = Path("data/generated")
OUTPUT_PATH = Path("data/all_methods_results.tsv")
DEFAULT_METHODS = ("teks_filter", "teks_filter_rag")
SUPPORTED_METHODS = (*DEFAULT_METHODS, "upper_bound_rag")
REQUIRED_COLUMNS = {"refusal_gt", "refusal_pred"}
METRIC_COLUMNS = ["acc", "prec", "rec", "refusal_f1", "allow_f1", "macro_f1"]
COLUMNS = [
    "method", "Model", "query_type", "corpus", "retriever", "split",
    "n_rows", "n_invalid", "status", "TP", "TN", "FP", "FN",
    *METRIC_COLUMNS, "file",
]


def get_metadata(method: str, file: Path, root: Path) -> dict:
    """Decode known folder layouts without assuming fixed parent depths."""
    parts = file.relative_to(root).parts[:-1]  # exclude outputs.jsonl
    model = parts[0] if len(parts) >= 1 else None
    query_type = parts[1] if len(parts) >= 2 else None
    extra = list(parts[2:])

    split = extra.pop() if extra and extra[-1].lower() in {"train", "dev", "test"} else None
    corpus = extra.pop(0) if extra and extra[0].lower() in {"full", "split"} else None
    retriever = "/".join(extra) if extra else None

    return {
        "method": method,
        "Model": model,
        "query_type": query_type.lower() if query_type else None,
        "corpus": corpus,
        "retriever": retriever,
        "split": split,
        "file": file.as_posix(),
    }


def read_predictions(file: Path):
    """Read JSONL line by line, tolerating an incomplete/malformed line."""
    records = []
    invalid = 0
    with file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("Expected a JSON object")
                records.append(record)
            except (ValueError, json.JSONDecodeError) as exc:
                invalid += 1
                print(f"WARNING: {file}:{line_number}: {exc}")
    return pd.DataFrame.from_records(records), invalid


def parse_bool(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    return values.map({"true": True, "false": False})


def evaluate_file(method: str, file: Path, root: Path) -> dict:
    row = {
        **get_metadata(method, file, root),
        "n_rows": 0,
        "n_invalid": 0,
        "status": "empty",
        **{col: pd.NA for col in ["TP", "TN", "FP", "FN", *METRIC_COLUMNS]},
    }
    try:
        df, bad_json = read_predictions(file)
    except OSError as exc:
        print(f"WARNING: Could not read {file}: {exc}")
        row["status"] = "read_error"
        return row

    if df.empty:
        row["n_invalid"] = bad_json
        row["status"] = "invalid" if bad_json else "empty"
        return row

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        row["status"] = "missing_columns"
        row["n_invalid"] = len(df) + bad_json
        print(f"WARNING: {file}: missing columns {sorted(missing)}")
        return row

    truth = parse_bool(df["refusal_gt"])
    predictions = parse_bool(df["refusal_pred"])
    valid = truth.notna() & predictions.notna()

    # Duplicate IDs are invalid for evaluation; keep the first valid occurrence.
    if "id" in df.columns:
        valid_ids = df.loc[valid, "id"]
        duplicate_ids = valid_ids.notna() & valid_ids.duplicated(keep="first")
        valid.loc[valid_ids.index[duplicate_ids]] = False

    row["n_rows"] = int(valid.sum())
    row["n_invalid"] = int((~valid).sum()) + bad_json
    if not row["n_rows"]:
        row["status"] = "invalid"
        return row

    y_true = truth[valid].astype(bool)
    y_pred = predictions[valid].astype(bool)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[False, True]).ravel()

    row.update({
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "acc": (tp + tn) / len(y_true),
        "prec": tp / (tp + fp) if tp + fp else 0.0,
        "rec": tp / (tp + fn) if tp + fn else 0.0,
        "refusal_f1": f1_score(y_true, y_pred, pos_label=True, zero_division=0),
        "allow_f1": f1_score(y_true, y_pred, pos_label=False, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "status": "partial_invalid" if row["n_invalid"] else "ok",
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS,
                        default=list(DEFAULT_METHODS))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    results = []
    for method in args.methods:
        root = GENERATED_DIR / f"{method}_results"
        if not root.is_dir():
            print(f"WARNING: Method directory does not exist: {root}")
            continue

        # Config files can exist before outputs.jsonl is created.
        files = sorted(set(root.rglob("outputs.jsonl")) | {
            path.with_name("outputs.jsonl")
            for path in root.rglob("outputs_config.json")
        })
        for file in files:
            if file.exists():
                row = evaluate_file(method, file, root)
            else:
                row = {
                    **get_metadata(method, file, root),
                    "n_rows": 0, "n_invalid": 0, "status": "no_outputs",
                    **{col: pd.NA for col in ["TP", "TN", "FP", "FN", *METRIC_COLUMNS]},
                }
            results.append(row)
            print(f"{method} | {row['Model']} | {row['query_type']} | "
                  f"{row['n_rows']:,} rows | {row['status']}")

    summary = pd.DataFrame(results, columns=COLUMNS)
    if not summary.empty:
        summary[METRIC_COLUMNS] = summary[METRIC_COLUMNS].round(3)
        summary = summary.sort_values(
            ["method", "Model", "query_type", "corpus", "retriever", "split"],
            na_position="last", kind="stable"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    summary.to_csv(temp_path, sep="\t", index=False)
    os.replace(temp_path, args.output)
    print(f"\nSaved {len(summary):,} experiment configurations to {args.output}")
    if not summary.empty:
        print(summary.drop(columns="file").to_string(index=False))


if __name__ == "__main__":
    main()
