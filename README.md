# LLMs Evolutionary Merging for L2S Math Reasoning

Project for **Deep Learning and Applied AI 2024/2025 (Sapienza University of Rome)**.  
We explore **evolutionary model merging** (via [Mergenetic](https://arxiv.org/abs/2505.11427)) we to combine:
- **DeepSeek-R1-Distill-Qwen** (slow-thinking, accurate Chain-of-Thought model),
- **Qwen2.5-Math** (fast-thinking, concise direct-answer model),

to improve the trade-off between **accuracy** and **response length** in **Long-to-Short (L2S) mathematical reasoning**.

---

## 📂 Repository Structure

```
LLMs Evolutionary Merging for L2S Math Reasoning
|
├── project_notebook.ipynb                   # step-by-step execution for merge and evaluation 
|
├── images/                                  # Figures for report
│
├── Qwen2.5-Math/                            # Evaluation framework directory
│   ├── README.md
│   └── evaluation/
│       ├── math_eval.py                     # Core evaluation function
│       ├── compute_NTR.py                   # Negative Transfer analysis
│       ├── generate_score_tensors.py        # Generate binary tensor for each benchmatk: 1=correct, 0=wrong answer
│       ├── extract_result.py                # Extract results from json/jsonl output files
│       ├── sh/
│       │   ├── l2s_eval.sh                  # Generic evaluation script → calls math_eval.py
│       │   └── qwen_eval.sh                 # Wrapper (set params → calls l2s_eval.sh)
│       ├── data/                            # Benchmarks (gsm8k, aime24, math500, minerva, etc.)
│       └── outputs/                         # Results (Baselines, Mergenetic, etc.)
│
└── mergenetic/                              # Evolutionary merging framework directory
    ├── README.md
    ├── requirements.txt
    ├── requirements_nb.txt
    ├── environment.yml
    ├── scripts/
    │   ├── run_mergekit.py                  # MergeKit utility function
    │   ├── mergenetic_gsm8k_TA.py           # end-to-end Task Arithmetic merge
    │   └── mergenetic_gsm8k_TIES.py         # end-to-end TIES merge
    ├── src/mergenetic/                      # Core library
    │   ├── merging/
    │   ├── optimization/
    │   ├── evaluation/
    │   ├── estimator/
    │   ├── searcher/
    │   └── utils.py
    ├── models/                              # folder to store DeepSeek, Qwen, merged models
    └── experiments/                         # Results of evolutionary runs (CSV logs, configs)
```

---

## 🛠️ Environment Setup

There is **no single global requirements file**. Each component has its own environment:

- `mergenetic/requirements.txt` → for merging  
- `Qwen2.5-Math/evaluation/requirements.txt` → for evaluation  


```bash
# 🐍 Create and activate the virtual environment
python3.11 -m venv ~/mergenetic/.venv
source ~/mergenetic/.venv/bin/activate

# 📦 Install all required dependencies 

## Install basic packages for notebooks:
pip install jupyter ipykernel
pip install --upgrade pip

## dependencies for Mergenetic framework
cd mergenetic
pip install -r requirements.txt
pip install -e .

## dependencies for Qwen2.5-Math evaluation framework
cd ../Qwen2.5-Math/evaluation
pip install -r requirements.txt
cd latex2sympy  
pip install -e . 
```

---

## 🚀 Usage

You can either run **scripts end-to-end** or follow the **notebooks** for step-by-step execution.

### 1. Merging (Task Arithmetic / TIES)

From `mergenetic/`:
**Task Arithmetic (TA)**
```bash
python mergenetic/scripts/mergenetic_gsm8k_TA.py 
```
**TIES**
```bash
python mergenetic/scripts/mergenetic_gsm8k_TIES.py
```
### 2. Evaluation

Two options:

**A) Shell scripts**  
- Edit `evaluation/sh/qwen_eval.sh` to set:
  - `MODEL_PATH` → path to the model checkpoint
  - `OUTPUT_DIR` → folder to store results
  - `DATASETS`, `SEED`, etc. as needed  
- Run:
  ```bash
  bash evaluation/sh/qwen_eval.sh
  ```
  Execution flow:  
  `qwen_eval.sh` → calls `l2s_eval.sh` → calls `math_eval.py`.

**B) Notebook**  
- Follow the Evaluation section of the notebook and run cells that call `math_eval.py` directly.  
- This is the easiest option if you want a **guided, step-by-step run**.

Results are stored under `Qwen2.5-Math/evaluation/outputs/.`.

---

## 📊 Metrics & Benchmarks

- **Accuracy**: Exact Match (on numerical answers)  
- **Output length**: average tokens generated  
- **Negative Transfer Rate (NTR)** (optional)  
- Benchmarks: `gsm8k`, `aime24`, `math`, `minerva_math`, `olympiadbench`, `college_math`

---

## 👤 Author

Mario Iacobelli — 1841427  
Sapienza University of Rome
