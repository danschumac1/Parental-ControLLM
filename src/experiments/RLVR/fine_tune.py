# python ./src/experiments/RLVR/fine_tune.py

import sys
sys.path.append("./src")

from utils.rlvr_utils import format_response
from utils.file_io import load_tsv_file

DATA_INPUT_FOLDER = "data/splits/"


def load_data() -> tuple[list[dict], list[dict], list[dict]]:
    train = load_tsv_file(DATA_INPUT_FOLDER + "train.tsv")
    dev = load_tsv_file(DATA_INPUT_FOLDER + "dev.tsv")
    test = load_tsv_file(DATA_INPUT_FOLDER + "test.tsv")

    return train, dev, test


def format_prompt(sample: dict) -> str:
    return (
        f"You are evaluating a question from a student in grade {sample['pretend_age']}.\n"
        f"Parental configuration: {sample['config']}\n"
        f"Student query: {sample['situation']}\n\n"
        "Identify the relevant educational expectations, determine age "
        "appropriateness, apply parental controls, and output ALLOW or BLOCK."
    )


def prepare_dataset(data: list[dict]) -> list[dict]:
    formatted = []

    for sample in data:
        response = format_response(
            expectations=[(sample["grade_span"], sample["expectation"])],
            pretend_age=int(sample["pretend_age"]),
            module=sample["module"],
            module_code=sample["module_code"],
            appropriateness=sample["pretend_age_type"],
            parent_config=sample["config"],
        )

        formatted.append({
            "prompt": format_prompt(sample),
            "completion": response,
        })

    return formatted


def main():
    train, dev, test = load_data()

    train = prepare_dataset(train)
    dev = prepare_dataset(dev)
    test = prepare_dataset(test)

    print(f"Train: {len(train):,}")
    print(f"Dev:   {len(dev):,}")
    print(f"Test:  {len(test):,}")

    print("\nExample prompt:\n", train[0]["prompt"])
    print("\nGround truth:\n", train[0]["completion"])


if __name__ == "__main__":
    main()