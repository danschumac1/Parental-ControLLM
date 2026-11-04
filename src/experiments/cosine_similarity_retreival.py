
"""
2026-11-04
Cosine similarity retrieval of HECAT expectations.

Run:
    python ./src/experiments/cosine_similarity_retreival.py
    python ./src/experiments/cosine_similarity_retreival.py --split train --n_r_contexts 30
    python ./src/experiments/cosine_similarity_retreival.py --split dev --query_col eduq
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data/splits")
OUTPUT_DIR = Path("data/retrieval")


def load_data(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{split}.tsv", sep="\t", dtype=str).fillna("")


def get_expectations(df: pd.DataFrame) -> pd.DataFrame:
    """Build a unique expectation corpus from the selected split."""
    columns = ["code", "module_code", "module", "grade_span", "expectation"]

    expectations = (
        df[columns]
        .drop_duplicates(subset=["code"])
        .reset_index(drop=True)
    )

    return expectations


def retrieve(
    queries: list[str],
    expectations: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int,
) -> list[list[dict]]:
    """Retrieve the top-K expectations for each query."""

    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    expectation_embeddings = model.encode(
        expectations["expectation"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    similarities = np.asarray(query_embeddings) @ np.asarray(expectation_embeddings).T

    results = []

    for scores in similarities:
        top_indices = np.argsort(-scores, kind="stable")[:top_k]

        retrieved = []

        for rank, idx in enumerate(top_indices, start=1):
            exp = expectations.iloc[int(idx)]

            retrieved.append({
                "rank": rank,
                "code": exp["code"],
                "module_code": exp["module_code"],
                "module": exp["module"],
                "grade_span": exp["grade_span"],
                "expectation": exp["expectation"],
                "similarity": float(scores[idx]),
            })

        results.append(retrieved)

    return results


def evaluate(
    ground_truth_codes: list[str],
    results: list[list[dict]],
    top_k: int,
) -> dict:
    """Calculate Recall@K and mean reciprocal rank."""

    ranks = []

    for gt_code, retrieved in zip(ground_truth_codes, results):
        rank = next(
            (item["rank"] for item in retrieved if item["code"] == gt_code),
            None,
        )
        ranks.append(rank)

    metrics = {
        f"recall@{k}": sum(
            rank is not None and rank <= k for rank in ranks
        ) / len(ranks)
        for k in [1, 5, 10, 20] if k <= top_k
    }

    metrics[f"mrr@{top_k}"] = sum(
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    ) / len(ranks)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        default="train",
    )
    parser.add_argument(
        "--query_col",
        choices=["situation", "eduq"],
        default="situation",
    )
    parser.add_argument(
        "--n_r_contexts",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    df = load_data(args.split)
    expectations = get_expectations(df)

    if args.n_r_contexts < 1:
        parser.error("--n_r_contexts must be at least 1")

    if df.empty or expectations.empty:
        parser.error("Selected split contains no queries or expectations")

    if (df[args.query_col] == "").any():
        parser.error(f"Some rows have empty {args.query_col} values")

    top_k = min(args.n_r_contexts, len(expectations))

    print(f"Split:        {args.split}")
    print(f"Query column: {args.query_col}")
    print(f"Queries:      {len(df):,}")
    print(f"Expectations: {len(expectations):,}")
    print(f"Top-K:        {top_k}")
    print(f"Model:        {args.model}")

    model = SentenceTransformer(args.model)

    results = retrieve(
        queries=df[args.query_col].tolist(),
        expectations=expectations,
        model=model,
        top_k=top_k,
    )

    metrics = evaluate(
        ground_truth_codes=df["code"].tolist(),
        results=results,
        top_k=top_k,
    )

    print("\nRetrieval Performance")
    for name, value in metrics.items():
        print(f"{name:<12} {value:.4f}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = df[["id", "code", args.query_col]].copy()
    output["retrieved_expectations"] = [
        json.dumps(items, ensure_ascii=False)
        for items in results
    ]

    output_path = OUTPUT_DIR / (
        f"{args.split}_{args.query_col}_top{top_k}.tsv"
    )

    output.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved: {output_path}")

    # Visual inspection
    print("\n" + "=" * 80)
    print("EXAMPLE RETRIEVAL")
    print(f"Query: {df.iloc[0][args.query_col]}")
    print(f"Ground truth: {df.iloc[0]['code']}")

    for item in results[0][:5]:
        marker = "*" if item["code"] == df.iloc[0]["code"] else " "
        print(
            f"{marker} {item['rank']:>2}. "
            f"[{item['code']}] "
            f"({item['similarity']:.3f}) "
            f"{item['expectation']}"
        )


if __name__ == "__main__":
    main()
