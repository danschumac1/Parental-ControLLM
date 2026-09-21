import json
import os
from datetime import datetime, timezone
from pathlib import Path


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_annotations(path):
    """Latest entry wins if duplicate sample_ids exist."""
    annotations = {}
    for row in load_jsonl(path):
        annotations[row["sample_id"]] = row
    return annotations


def save_annotation(sample, aligned, path, annotator):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(path)
    annotations[sample["sample_id"]] = {
        **sample,
        "aligned": aligned,
        "annotator_id": annotator,
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    rows = sorted(annotations.values(), key=lambda x: x["pair_index"])

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    os.replace(tmp, path)


def first_unannotated(samples, annotations):
    for i, row in enumerate(samples):
        if row["sample_id"] not in annotations:
            return i
    return None