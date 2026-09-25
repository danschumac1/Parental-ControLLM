'''
python ./src/experiments/agreement.py
'''

import os
import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ANNOTATION_DIR = "./data/annotations/situation_alignment"

ANNOTATOR_PAIRS = [
    ("dan", "harley"),
    ("urmi", "visha"),
]


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return pd.DataFrame(rows)


def evaluate_pair(
    annotator_1: str,
    annotator_2: str,
) -> dict:

    path_1 = os.path.join(ANNOTATION_DIR, f"{annotator_1}.jsonl")
    path_2 = os.path.join(ANNOTATION_DIR, f"{annotator_2}.jsonl")

    df_1 = load_jsonl(path_1)
    df_2 = load_jsonl(path_2)

    # Keep expectation/situation from annotator 1,
    # plus each annotator's aligned label.
    df_1 = df_1[
        [
            "sample_id",
            "expectation",
            "situation",
            "aligned",
        ]
    ].rename(
        columns={
            "aligned": f"{annotator_1}_aligned"
        }
    )

    df_2 = df_2[
        [
            "sample_id",
            "aligned",
        ]
    ].rename(
        columns={
            "aligned": f"{annotator_2}_aligned"
        }
    )

    # Only evaluate samples annotated by both annotators.
    paired = df_1.merge(
        df_2,
        on="sample_id",
        how="inner",
    )

    labels_1 = paired[f"{annotator_1}_aligned"]
    labels_2 = paired[f"{annotator_2}_aligned"]

    agreement = (labels_1 == labels_2).mean()
    kappa = cohen_kappa_score(labels_1, labels_2)

    # Keep only disagreements.
    disagreements = paired[
        labels_1 != labels_2
    ].copy()

    disagreement_columns = [
        "sample_id",
        "expectation",
        "situation",
        f"{annotator_1}_aligned",
        f"{annotator_2}_aligned",
    ]

    disagreements = disagreements[disagreement_columns]

    output_path = os.path.join(
        ANNOTATION_DIR,
        f"{annotator_1}_{annotator_2}_disagreements.tsv",
    )

    disagreements.to_csv(
        output_path,
        sep="\t",
        index=False,
    )

    return {
        "pair": f"{annotator_1}-{annotator_2}",
        "n": len(paired),
        "n_agree": (labels_1 == labels_2).sum(),
        "n_disagree": (labels_1 != labels_2).sum(),
        "agreement": agreement,
        "cohen_kappa": kappa,
        "disagreement_file": output_path,
    }


def main() -> None:

    results = []

    for annotator_1, annotator_2 in ANNOTATOR_PAIRS:
        result = evaluate_pair(
            annotator_1=annotator_1,
            annotator_2=annotator_2,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print()
    print("INTER-ANNOTATOR AGREEMENT")
    print("==========================")
    print()

    for _, row in results_df.iterrows():

        print(row["pair"])
        print(f"N:               {row['n']}")
        print(f"Agreements:      {row['n_agree']}")
        print(f"Disagreements:   {row['n_disagree']}")
        print(f"Agreement:       {row['agreement']:.2%}")
        print(f"Cohen's kappa:   {row['cohen_kappa']:.4f}")
        print(f"Saved:           {row['disagreement_file']}")
        print()


if __name__ == "__main__":
    main()