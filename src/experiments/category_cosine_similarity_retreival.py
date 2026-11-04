
"""
2026-11-04
Two-stage cosine similarity retrieval:
    1. Query -> Top-N most similar HECAT categories
    2. Query -> Top-K expectations across selected categories

Run:
    python ./src/experiments/category_cosine_similarity_retreival.py
    python ./src/experiments/category_cosine_similarity_retreival.py --split train --n_categories 2 --n_r_contexts 20
    python ./src/experiments/category_cosine_similarity_retreival.py --split train --query_col eduq --n_categories 2
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
    return pd.read_csv(
        DATA_DIR / f"{split}.tsv",
        sep="\t",
        dtype=str,
    ).fillna("")


def get_expectations(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "code", "module_code", "module",
        "grade_span", "expectation",
    ]

    return (
        df[columns]
        .drop_duplicates(subset=["code"])
        .reset_index(drop=True)
    )


def retrieve(
    queries: list[str],
    expectations: pd.DataFrame,
    model: SentenceTransformer,
    top_k: int,
    n_categories: int,
) -> tuple[list[list[str]], list[list[dict]]]:
    """Predict Top-N categories, then retrieve Top-K expectations."""

    # Unique categories available in this split
    categories = (
        expectations[["module_code", "module"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Embed queries, categories, and expectations
    query_embeddings = np.asarray(model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
    ))

    category_embeddings = np.asarray(model.encode(
        categories["module"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ))

    expectation_embeddings = np.asarray(model.encode(
        expectations["expectation"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True,
    ))

    # ---------------------------------------------------------
    # Stage 1: Query -> Top-N categories
    # ---------------------------------------------------------
    category_similarities = query_embeddings @ category_embeddings.T

    top_category_indices = np.argsort(
        -category_similarities,
        axis=1,
        kind="stable",
    )[:, :n_categories]

    predicted_categories = [
        categories.iloc[indices]["module_code"].tolist()
        for indices in top_category_indices
    ]

    # ---------------------------------------------------------
    # Stage 2: Query -> Top-K expectations within categories
    # ---------------------------------------------------------
    results = []

    module_codes = expectations["module_code"].to_numpy()

    for query_embedding, selected_categories in zip(
        query_embeddings, predicted_categories
    ):
        category_indices = np.flatnonzero(
            np.isin(module_codes, selected_categories)
        )

        scores = expectation_embeddings[category_indices] @ query_embedding

        top_indices = np.argsort(
            -scores,
            kind="stable",
        )[:top_k]

        retrieved = []

        for rank, local_idx in enumerate(top_indices, start=1):
            idx = int(category_indices[local_idx])
            exp = expectations.iloc[idx]

            retrieved.append({
                "rank": rank,
                "code": exp["code"],
                "module_code": exp["module_code"],
                "module": exp["module"],
                "grade_span": exp["grade_span"],
                "expectation": exp["expectation"],
                "similarity": float(scores[local_idx]),
            })

        results.append(retrieved)

    return predicted_categories, results


def evaluate(
    ground_truth_codes: list[str],
    ground_truth_categories: list[str],
    predicted_categories: list[list[str]],
    results: list[list[dict]],
    top_k: int,
) -> dict:
    """Evaluate category Recall@N, expectation Recall@K, and MRR."""

    n = len(results)

    # Category recall: ground truth category in selected Top-N
    category_recall = sum(
        gt in pred
        for pred, gt in zip(predicted_categories, ground_truth_categories)
    ) / n

    # Ground-truth expectation rank
    ranks = []

    for gt_code, retrieved in zip(ground_truth_codes, results):
        rank = next(
            (item["rank"] for item in retrieved if item["code"] == gt_code),
            None,
        )
        ranks.append(rank)

    metrics = {"category_recall": category_recall}

    for k in [1, 5, 10, 20]:
        if k <= top_k:
            metrics[f"recall@{k}"] = sum(
                rank is not None and rank <= k
                for rank in ranks
            ) / n

    metrics[f"mrr@{top_k}"] = sum(
        1.0 / rank if rank is not None else 0.0
        for rank in ranks
    ) / n

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
        "--n_categories",
        type=int,
        default=10,
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

    if df.empty or expectations.empty:
        parser.error("Selected split contains no queries or expectations")

    if args.n_categories < 1:
        parser.error("--n_categories must be at least 1")

    if args.n_r_contexts < 1:
        parser.error("--n_r_contexts must be at least 1")

    if (df[args.query_col] == "").any():
        parser.error(f"Some rows have empty {args.query_col} values")

    n_categories = min(
        args.n_categories,
        expectations["module_code"].nunique(),
    )

    top_k = min(args.n_r_contexts, len(expectations))

    print(f"Split:        {args.split}")
    print(f"Query column: {args.query_col}")
    print(f"Queries:      {len(df):,}")
    print(f"Expectations: {len(expectations):,}")
    print(f"Categories:   {expectations['module_code'].nunique()}")
    print(f"Top-N cats:   {n_categories}")
    print(f"Top-K exps:   {top_k}")
    print(f"Model:        {args.model}")

    model = SentenceTransformer(args.model)

    predicted_categories, results = retrieve(
        queries=df[args.query_col].tolist(),
        expectations=expectations,
        model=model,
        top_k=top_k,
        n_categories=n_categories,
    )

    metrics = evaluate(
        ground_truth_codes=df["code"].tolist(),
        ground_truth_categories=df["module_code"].tolist(),
        predicted_categories=predicted_categories,
        results=results,
        top_k=top_k,
    )

    print("\nRetrieval Performance")
    for name, value in metrics.items():
        print(f"{name:<20} {value:.4f}")

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = df[["id", "code", "module_code", args.query_col]].copy()

    output["predicted_module_codes"] = [
        json.dumps(cats) for cats in predicted_categories
    ]

    output["retrieved_expectations"] = [
        json.dumps(items, ensure_ascii=False)
        for items in results
    ]

    output_path = OUTPUT_DIR / (
        f"{args.split}_{args.query_col}_category{n_categories}_top{top_k}.tsv"
    )

    output.to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved: {output_path}")

    # ---------------------------------------------------------
    # Visual inspection
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXAMPLE RETRIEVAL")
    print(f"Query: {df.iloc[0][args.query_col]}")
    print(f"Ground-truth category: {df.iloc[0]['module_code']}")
    print(f"Predicted categories:  {predicted_categories[0]}")
    print(f"Ground-truth expectation: {df.iloc[0]['code']}")

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
