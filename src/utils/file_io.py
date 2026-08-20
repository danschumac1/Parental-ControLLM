import os
import yaml
import json



def load_yaml_prompt(prompt_path):
    """Load a YAML prompt template from the specified path."""
    with open(prompt_path, "r") as f:
        return yaml.safe_load(f)


def load_yaml_prompts(prompt_types, prompt_folder):
    prompts = {}
    for name in prompt_types:
        with open(f"{prompt_folder}/{name}.yaml", "r") as f:
            prompts[name] = yaml.safe_load(f)
    return prompts


def load_single_yaml_prompt(prompt_path):
    """Load a single YAML prompt template from the specified path."""
    with open(prompt_path, "r") as f:
        return yaml.safe_load(f)


def load_tsv_file(file_path):
    """Read a TSV file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split('\t')
        data = []
        for line in lines[1:]:
            values = line.strip().split('\t')
            data.append(dict(zip(headers, values)))
        return data


def load_jsonl_file(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

    
def append_jsonl_file(file_path, data):
    """Append a list of dictionaries to a JSONL file."""

    with open(file_path, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_jsonl_file(file_path, data):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')


def remove_completed_items(
    data,
    output_path,
    idx_key="item_number",
):
    if not os.path.exists(output_path):
        print(f"{len(data)} items remaining.")
        return data

    completed_items = set()

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                completed_items.add(
                    json.loads(line)[idx_key]
                )

    if completed_items:
        print(f"Found {len(completed_items)} completed items.")
        print("Removing completed items from this run.")

    data = [
        row
        for row in data
        if row[idx_key] not in completed_items
    ]

    if not data:
        print("All requested items have already been completed.")
    else:
        print(f"{len(data)} items remaining.")

    return data

    return completed_items


def save_args_as_config_json(args, output_path):
    config = vars(args)

    config_path = output_path.replace(
        ".jsonl",
        "_config.json",
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=4,
        )
