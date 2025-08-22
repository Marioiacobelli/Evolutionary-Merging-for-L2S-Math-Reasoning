import os
import json
from transformers import AutoTokenizer

def compute_avg_token_length_from_code(jsonl_path, tokenizer):
    total_length = 0
    total_items = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            code = data.get("code", [])
            if code:
                text = code[0] if isinstance(code, list) else code
                tokenized = tokenizer(text, return_tensors="pt")
                total_length += tokenized.input_ids.shape[1]
                total_items += 1

    if total_items == 0:
        return 0.0
    return total_length / total_items

def extract_accuracy(json_path):
    if os.path.isfile(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("acc", None)
            except json.JSONDecodeError:
                return None
    return None

def analyze_all_tasks(root_folder, tokenizer):
    results = []

    total_acc = 0.0
    total_len = 0.0
    count_acc = 0
    count_len = 0

    for task_name in sorted(os.listdir(root_folder)):
        task_path = os.path.join(root_folder, task_name)
        if not os.path.isdir(task_path):
            continue

        jsonl_files = [f for f in os.listdir(task_path) if f.endswith(".jsonl")]
        acc_file = [f for f in os.listdir(task_path) if f.endswith(".json")]

        if not jsonl_files:
            continue

        jsonl_path = os.path.join(task_path, jsonl_files[0])
        acc_path = os.path.join(task_path, acc_file[0]) if acc_file else None

        avg_len = compute_avg_token_length_from_code(jsonl_path, tokenizer)
        acc = extract_accuracy(acc_path) if acc_path else None

        if avg_len > 0:
            total_len += avg_len
            count_len += 1
        if acc is not None:
            total_acc += acc
            count_acc += 1

        results.append((task_name, acc, avg_len))

    print("\n==== Summary: Average Token Lengths and Accuracy per Task ====")
    print(f"{'Task':<25} {'Accuracy':<10} {'Avg Token Len':<15}")
    print("-" * 55)
    for task, acc, avg_len in results:
        acc_str = f"{acc:.1f}" if acc is not None else "N/A"
        print(f"{task:<25} {acc_str:<10} {avg_len:.2f}")

    avg_acc_overall = total_acc / count_acc if count_acc > 0 else 0.0
    avg_len_overall = total_len / count_len if count_len > 0 else 0.0

    print("\n==== Overall Averages ====")
    print(f"{'Avg Accuracy':<25}: {avg_acc_overall:.1f}")
    print(f"{'Avg Token Length':<25}: {avg_len_overall:.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_folder", type=str, required=True, help="Root folder with task subdirectories")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Tokenizer model name or path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    analyze_all_tasks(args.root_folder, tokenizer)