"""
2026-09-18

Granular analysis of guardrail evaluation results.

Run:
    python ./src/eval_guardrail.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.file_io import load_jsonl_file


IN_PATH = "data/generated/guardrail_results/gpt-5.4-nano/outputs.jsonl"

OUT_DIR = Path(
    "data/generated/guardrail_results/gpt-5.4-nano/analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# DATA PREPARATION
# =========================================================

def parse_blocked_cats(value):
    """
    Turn blocked_cats into a Python list.

    Example:
        "AOD,FN,MEH" -> ["AOD", "FN", "MEH"]
        ""           -> []
    """

    if not value:
        return []

    return [
        cat.strip()
        for cat in str(value).split(",")
        if cat.strip()
    ]


def prepare_dataframe(data):
    """
    Convert JSONL rows into a DataFrame and add useful
    derived columns.
    """

    df = pd.DataFrame(data)

    # Convert blocked category string into a list
    df["blocked_cat_list"] = (
        df["blocked_cats"]
        .fillna("")
        .apply(parse_blocked_cats)
    )

    # Number of blocked categories for this configuration
    df["num_blocked"] = (
        df["blocked_cat_list"]
        .apply(len)
    )

    # Is THIS question's category blocked?
    df["target_blocked"] = df.apply(
        lambda row:
            row["module_code"] in row["blocked_cat_list"],
        axis=1,
    )

    # Make error type explicit
    conditions = [
        (df["refusal_gt"] == True)
        & (df["refusal_pred"] == True),

        (df["refusal_gt"] == False)
        & (df["refusal_pred"] == False),

        (df["refusal_gt"] == False)
        & (df["refusal_pred"] == True),

        (df["refusal_gt"] == True)
        & (df["refusal_pred"] == False),
    ]

    labels = [
        "TP",
        "TN",
        "FP",
        "FN",
    ]

    df["error_type"] = np.select(
        conditions,
        labels,
        default="UNKNOWN",
    )

    df["num_blocked"] = (
    df["config"]
    .str.count("x")
    )

    df["num_allowed"] = (
        df["config"]
        .str.count("o")
    )

    return df


# =========================================================
# METRICS
# =========================================================

def safe_divide(numerator, denominator):
    """
    Return NaN when the metric is undefined instead of 0.

    This is useful for groups like all_allowed where
    there may be no positive examples.
    """

    if denominator == 0:
        return np.nan

    return numerator / denominator


def calculate_metrics(df):
    """
    Calculate metrics for one subset.

    Classes:
        True  = refusal
        False = allowed
    """

    tp = (
        (df["refusal_gt"] == True)
        & (df["refusal_pred"] == True)
    ).sum()

    tn = (
        (df["refusal_gt"] == False)
        & (df["refusal_pred"] == False)
    ).sum()

    fp = (
        (df["refusal_gt"] == False)
        & (df["refusal_pred"] == True)
    ).sum()

    fn = (
        (df["refusal_gt"] == True)
        & (df["refusal_pred"] == False)
    ).sum()

    total = len(df)

    accuracy = safe_divide(
        tp + tn,
        total,
    )

    # -----------------------------------------------------
    # REFUSAL CLASS
    # -----------------------------------------------------

    refusal_precision = safe_divide(
        tp,
        tp + fp,
    )

    refusal_recall = safe_divide(
        tp,
        tp + fn,
    )

    refusal_f1 = safe_divide(
        2 * tp,
        2 * tp + fp + fn,
    )

    # -----------------------------------------------------
    # ALLOWED CLASS
    #
    # Treat "allowed" as the positive class:
    #
    # TP_allowed = TN
    # FP_allowed = FN
    # FN_allowed = FP
    # -----------------------------------------------------

    allowed_f1 = safe_divide(
        2 * tn,
        2 * tn + fp + fn,
    )

    # Only report macro-F1 when both classes are present
    # in the ground truth.
    has_refusal = (df["refusal_gt"] == True).any()
    has_allowed = (df["refusal_gt"] == False).any()

    if has_refusal and has_allowed:
        macro_f1 = (
            refusal_f1 + allowed_f1
        ) / 2
    else:
        macro_f1 = np.nan

    return {
        "n": total,

        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,

        "accuracy": accuracy,

        "refusal_precision": refusal_precision,
        "refusal_recall": refusal_recall,

        "refusal_f1": refusal_f1,
        "allowed_f1": allowed_f1,
        "macro_f1": macro_f1,

        "false_refusal_rate":
            safe_divide(fp, fp + tn),

        "false_allow_rate":
            safe_divide(fn, fn + tp),

        "predicted_refusal_rate":
            safe_divide(tp + fp, total),

        "expected_refusal_rate":
            safe_divide(tp + fn, total),
    }
# =========================================================
# GROUPED ANALYSIS
# =========================================================

def evaluate_by(df, column):
    """
    Calculate metrics separately for every value
    of a column.
    """

    results = []

    for value, group in df.groupby(
        column,
        dropna=False,
    ):
        metrics = calculate_metrics(group)

        metrics[column] = value

        results.append(metrics)

    result_df = pd.DataFrame(results)

    # Put grouping variable first
    columns = [
        column,
        *[
            col
            for col in result_df.columns
            if col != column
        ],
    ]

    return result_df[columns]


def print_analysis(title, results):
    """
    Print grouped results cleanly.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


# =========================================================
# HEAT MAP
# =========================================================

def make_accuracy_heatmap(
    df,
    row_variable,
    column_variable,
    filename,
):
    """
    Create an accuracy heat map for two grouping variables.

    Example:
        module_code × config_type
    """

    matrix = (
        df.groupby(
            [row_variable, column_variable]
        )["correct"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    image = ax.imshow(
        matrix.values,
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(
        range(len(matrix.columns))
    )

    ax.set_xticklabels(
        matrix.columns,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(matrix.index))
    )

    ax.set_yticklabels(
        matrix.index
    )

    ax.set_xlabel(
        column_variable
    )

    ax.set_ylabel(
        row_variable
    )

    ax.set_title(
        f"Accuracy by {row_variable} and {column_variable}"
    )

    # Put values inside cells
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):

            value = matrix.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Accuracy"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / filename,
        dpi=300,
    )

    plt.close(fig)


# =========================================================
# SANITY CHECKS
# =========================================================

def run_sanity_checks(df):
    """
    Check whether refusal_gt agrees with whether the
    item's category is actually blocked.
    """

    mismatches = df[
        df["target_blocked"]
        != df["refusal_gt"]
    ]

    print()
    print("SANITY CHECK")
    print("-" * 40)

    print(
        "Rows where target_blocked != refusal_gt:",
        len(mismatches),
    )

    if len(mismatches) > 0:

        print(
            mismatches[
                [
                    "id",
                    "module_code",
                    "config_type",
                    "blocked_cats",
                    "target_blocked",
                    "refusal_gt",
                ]
            ].head(20)
        )


def plot_random_accuracy_by_count(
    df,
    count_column,
    x_label,
    filename,
):
    """
    Compare accuracy for:
        - random_question_blocked
        - random_question_allowed

    grouped by the number of blocked/allowed categories.
    """

    random_df = df[
        df["config_type"].isin([
            "random_question_blocked",
            "random_question_allowed",
        ])
    ]

    results = (
        random_df
        .groupby(
            [
                "config_type",
                count_column,
            ]
        )
        .agg(
            accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
        .reset_index()
        .sort_values(
            [
                "config_type",
                count_column,
            ]
        )
    )

    print()
    print("=" * 70)
    print(f"ACCURACY BY {count_column.upper()}")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    # -----------------------------------------------------
    # PLOT
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    for config_type, group in results.groupby(
        "config_type"
    ):

        ax.plot(
            group[count_column],
            group["accuracy"],
            marker="o",
            label=config_type,
        )

    ax.set_xlabel(
        x_label
    )

    ax.set_ylabel(
        "Accuracy"
    )

    ax.set_title(
        f"Accuracy by {x_label}"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.grid(
        alpha=0.3,
    )

    ax.legend(
        title="Configuration"
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR / filename,
        dpi=300,
    )

    plt.close(fig)

# =========================================================
# MAIN
# =========================================================

def main():

    data = load_jsonl_file(
        IN_PATH
    )

    df = prepare_dataframe(
        data
    )

    print(
        f"Loaded {len(df)} rows"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    # -----------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------

    run_sanity_checks(
        df
    )

    # -----------------------------------------------------
    # OVERALL
    # -----------------------------------------------------

    overall = calculate_metrics(
        df
    )

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    for key, value in overall.items():

        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

        else:
            print(f"{key}: {value}")

    # -----------------------------------------------------
    # CATEGORY
    # -----------------------------------------------------

    by_category = evaluate_by(
        df,
        "module_code",
    )

    print_analysis(
        "BY CATEGORY",
        by_category,
    )

    by_category.to_csv(
        OUT_DIR / "by_category.csv",
        index=False,
    )

    # -----------------------------------------------------
    # CONFIGURATION TYPE
    # -----------------------------------------------------

    by_config = evaluate_by(
        df,
        "config_type",
    )

    print_analysis(
        "BY CONFIG TYPE",
        by_config,
    )

    by_config.to_csv(
        OUT_DIR / "by_config_type.csv",
        index=False,
    )

    # -----------------------------------------------------
    # GRADE SPAN
    # -----------------------------------------------------

    by_grade = evaluate_by(
        df,
        "grade_span",
    )

    print_analysis(
        "BY GRADE SPAN",
        by_grade,
    )

    by_grade.to_csv(
        OUT_DIR / "by_grade_span.csv",
        index=False,
    )

    # -----------------------------------------------------
    # EXPECTATION TYPE
    # -----------------------------------------------------

    by_expectation = evaluate_by(
        df,
        "expectation_type",
    )

    print_analysis(
        "BY EXPECTATION TYPE",
        by_expectation,
    )

    by_expectation.to_csv(
        OUT_DIR / "by_expectation_type.csv",
        index=False,
    )

    # -----------------------------------------------------
    # NUMBER OF BLOCKED CATEGORIES
    # -----------------------------------------------------

    by_num_blocked = evaluate_by(
        df,
        "num_blocked",
    )

    print_analysis(
        "BY NUMBER OF BLOCKED CATEGORIES",
        by_num_blocked,
    )

    by_num_blocked.to_csv(
        OUT_DIR / "by_num_blocked.csv",
        index=False,
    )

    # -----------------------------------------------------
    # HEAT MAPS
    # -----------------------------------------------------

    make_accuracy_heatmap(
        df=df,
        row_variable="module_code",
        column_variable="config_type",
        filename="category_by_config_accuracy.png",
    )

    make_accuracy_heatmap(
        df=df,
        row_variable="grade_span",
        column_variable="config_type",
        filename="grade_by_config_accuracy.png",
    )

    print()
    print(
        f"Analysis saved to: {OUT_DIR}"
    )

    plot_random_accuracy_by_count(
        df=df,
        count_column="num_blocked",
        x_label="Number of Blocked Categories",
        filename="random_accuracy_by_num_blocked.png",
    )

if __name__ == "__main__":
    main()