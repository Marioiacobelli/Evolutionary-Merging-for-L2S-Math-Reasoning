import yaml
import torch
import mergekit
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge
import sys

def merge_model_from_config(config_path: str, output_path: str):
    print(f"🚀 Start merge with config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    cfg = MergeConfiguration.model_validate(config_data)

    options = MergeOptions(
        cuda=torch.cuda.is_available(),
        copy_tokenizer=True,
        lazy_unpickle=True,
        low_cpu_memory=True,
    )

    print(f"⚙️  Merge options: {options}")
    run_merge(cfg, output_path, options=options)
    print(f"✅ Merge completed. Model saved in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_mergekit.py <config.yaml> <output_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2]

    merge_model_from_config(config_path, output_path)