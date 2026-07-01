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


def load_jsonl_file(file_path):
    """Read a JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def append_jsonl_file(file_path, data):
    """Append a list of dictionaries to a JSONL file."""
    with open(file_path, 'a') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

def save_jsonl_file(file_path, data):
    """Save a list of dictionaries to a JSONL file."""
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

