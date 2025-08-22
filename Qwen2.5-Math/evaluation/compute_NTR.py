import os
import sys
import torch
import pandas as pd
import argparse

def compute_negative_transfer(base1_dir, base2_dir, merged_dir):
    base1_files = {f.replace("_scores.pt", ""): os.path.join(base1_dir, f)
                   for f in os.listdir(base1_dir) if f.endswith(".pt")}
    base2_files = {f.replace("_scores.pt", ""): os.path.join(base2_dir, f)
                   for f in os.listdir(base2_dir) if f.endswith(".pt")}
    merged_files = {f.replace("_scores.pt", ""): os.path.join(merged_dir, f)
                    for f in os.listdir(merged_dir) if f.endswith(".pt")}

    common_tasks = set(base1_files) & set(base2_files) & set(merged_files)

    task_results = {}
    total_neg = 0
    total_eligible = 0
    total_all = 0

    for task in sorted(common_tasks):
        base1 = torch.load(base1_files[task]).bool()
        base2 = torch.load(base2_files[task]).bool()
        merged = torch.load(merged_files[task]).bool()

        if not (len(base1) == len(base2) == len(merged)):
            print(f"❌ Length mismatch in task {task}")
            continue

        correct_by_base = base1 | base2
        wrong_by_merged = ~merged
        neg_transfer = correct_by_base & wrong_by_merged

        num_neg = int(neg_transfer.sum().item())
        denom = int(correct_by_base.sum().item())
        total = merged.shape[0]
        ntr_task = num_neg / denom if denom > 0 else 0.0

        task_results[task] = {
            "negative_transfer_rate": ntr_task,
            "neg_transfer": num_neg,
            "eligible_examples": denom,
            "total_examples": total,
        }

        total_neg += num_neg
        total_eligible += denom
        total_all += total

    total_ntr = total_neg / total_eligible if total_eligible > 0 else 0.0

    df = pd.DataFrame.from_dict(task_results, orient="index")
    df.index.name = "task"

    return df, total_ntr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Negative Transfer Rate from score tensors.")
    parser.add_argument("--base1_dir", type=str, required=True, help="Path to base model 1 score tensors")
    parser.add_argument("--base2_dir", type=str, required=True, help="Path to base model 2 score tensors")
    parser.add_argument("--merged_dir", type=str, required=True, help="Path to merged model score tensors")

    args = parser.parse_args()

    df, ntr_value = compute_negative_transfer(args.base1_dir, args.base2_dir, args.merged_dir)

    print("\n📊 NTR per task:\n")
    print(df.to_markdown())
    print(f"\n✅ Aggregated NTR: {ntr_value:.4f}")