
MODULE_ORDER = ["AOD", "FN", "MEH", "PA", "PHW", "SH", "S", "T", "V"]


def format_response(
    expectations: list[tuple[str, str]],
    pretend_age: int,
    module: str,
    module_code: str,
    appropriateness: str,
    parent_config: str,
) -> str:
    # Expectations: [(grade_span, expectation), ...]
    expectation_lines = [
        f"- By grade {grade_span.split('-')[0]}, students should be able to {expectation}"
        for grade_span, expectation in expectations
    ]

    expectation_reasoning = (
        "This query most closely aligns with the following expectations:\n"
        + "\n".join(expectation_lines)
    )

    # Age appropriateness (ground-truth label)
    age_appropriate = appropriateness == "appropriate"

    age_reasoning = (
        f"The student is in grade {pretend_age}. "
        f"{'The query is age appropriate.' if age_appropriate else 'The query is not age appropriate.'}"
    )

    # Parental controls
    module_idx = MODULE_ORDER.index(module_code)
    parent_allowed = parent_config[module_idx] == "o"

    parental_reasoning = (
        f"This content has to do with {module}, "
        f"which the parent has "
        f"{'allowed' if parent_allowed else 'disallowed'}."
    )

    # Final decision
    decision = "ALLOW" if age_appropriate and parent_allowed else "BLOCK"

    return (
        f"<expectation_reasoning>\n{expectation_reasoning}\n</expectation_reasoning>\n"
        f"<age_reasoning>\n{age_reasoning}\n</age_reasoning>\n"
        f"<parental_control_reasoning>\n{parental_reasoning}\n</parental_control_reasoning>\n"
        f"<decision>{decision}</decision>"
    )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/splits/dev.tsv", sep="\t")

    for _, row in df.head(5).iterrows():
        print(f"\n{'=' * 80}")
        print(f"ID: {row['id']}")
        print(
            format_response(
                expectations=[(row["grade_span"], row["expectation"])],
                pretend_age=int(row["pretend_age"]),
                module=row["module"],
                module_code=row["module_code"],
                appropriateness=row["pretend_age_type"],
                parent_config=row["config"],
            )
        )
