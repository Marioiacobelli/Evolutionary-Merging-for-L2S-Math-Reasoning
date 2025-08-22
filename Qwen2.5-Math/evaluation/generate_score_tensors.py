import os
import json
import torch
import argparse

def generate_score_tensors(base_dir):
    output_dir = os.path.join(base_dir, "scores")
    os.makedirs(output_dir, exist_ok=True)

    for task_name in os.listdir(base_dir):
        task_path = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_path):
            continue

        jsonl_file = next((f for f in os.listdir(task_path) if f.endswith(".jsonl")), None)
        if not jsonl_file:
            continue

        jsonl_path = os.path.join(task_path, jsonl_file)

        try:
            scores = []
            with open(jsonl_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    score = obj.get("score", [False])[0]
                    scores.append(int(score))

            tensor = torch.tensor(scores, dtype=torch.uint8)

            output_file = os.path.join(output_dir, f"{task_name}_scores.pt")
            torch.save(tensor, output_file)

            print(f"✅ Saved: {output_file} ({len(scores)} samples)")
        except Exception as e:
            print(f"❌ Error in file {jsonl_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary score tensors from jsonl evaluation results.")
    parser.add_argument(
        "--base_dir", type=str, required=True,
        help="Path to directory containing per-task subfolders with .jsonl outputs"
    )
    args = parser.parse_args()

    generate_score_tensors(args.base_dir)