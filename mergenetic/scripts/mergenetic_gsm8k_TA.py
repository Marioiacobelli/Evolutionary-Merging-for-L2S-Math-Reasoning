# ========== Imports ==========
import sys, importlib, pathlib, os
import numpy as np
import yaml
import textwrap
import inspect

from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from mergenetic.searcher import Searcher
from mergenetic.utils import ConfigLmEval
from mergenetic.optimization.predefined_problems import MathReasoningProblem, ConfigPE
from mergenetic import PROJECT_ROOT
from mergenetic.merging.taskarithmetic_merger import TaskArithmeticMerger
from mergenetic import PROJECT_ROOT

from lm_eval.tasks import TaskManager
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# ========== Setup environment ==========
real_pkg_dir = pathlib.Path(f"{PROJECT_ROOT}/mergenetic/src/mergenetic")
if not real_pkg_dir.exists():
    raise RuntimeError("mergenetic/src/mergenetic not found – is the repo cloned?")

# Clean previously loaded modules
for name in list(sys.modules):
    if name == "mergenetic" or name.startswith("mergenetic."):
        del sys.modules[name]

# Setup sys.path
src_root = str(real_pkg_dir.parent)  # /mergenetic/src
if src_root not in sys.path:
    sys.path.insert(0, src_root)

for bad in ("", "/content"):
    if bad in sys.path:
        sys.path.remove(bad)

importlib.invalidate_caches()
import mergenetic
print("✅ Loaded mergenetic from:", getattr(mergenetic, '__file__', 'N/A'))



# ========== Model setup ==========
model_dir = f"{PROJECT_ROOT}/mergenetic/models"
os.makedirs(model_dir, exist_ok=True)

# DeepSeek-R1-Distill-Qwen-1.5B (Base model)
deepseek_repo = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
snapshot_download(
    repo_id=deepseek_repo,
    local_dir=os.path.join(model_dir, "DeepSeek-R1-Distill-Qwen-1.5B")
)

# Qwen2.5-Math-1.5B (Target model)
qwen_math_repo = "Qwen/Qwen2.5-Math-1.5B"
snapshot_download(
    repo_id=qwen_math_repo,
    local_dir=os.path.join(model_dir, "Qwen2.5-Math-1.5B")
)

# ========== Config ==========
config = ConfigLmEval()
config.additional_templates_folder = os.path.join(PROJECT_ROOT, "mergenetic", "lm_tasks")
config.bench = "gsm8k"

task_name = "gsm8k_cot_zeroshot_new"
lang_id = "en"

task_manager = TaskManager(include_path=config.additional_templates_folder)
task = task_manager.load_task_or_group(task_name)[task_name]

config.n_samples = 30
num_test_samples = len(task.dataset["test"])

anchors = np.random.choice(range(num_test_samples), config.n_samples, replace=False)
anchors_weights = np.ones(len(anchors)) / len(anchors)

config.seed = 42
config.device = "cuda"
config.run_id = "TA_multiobjective_1.5B"
config.tasks = {"search": {"en": task_name}, "test": {"en": task_name}}
config.metric = "exact_match"
config.task_type = "FG_MATH"
config.path_to_store_config = f"{PROJECT_ROOT}/mergenetic/experiments/evolutionary-merging-lm-harness"
config.path_to_store_merged_model = f"{model_dir}/merged"
config.base_model = f"{model_dir}/DeepSeek-R1-Distill-Qwen-1.5B"
config.models = {"en": f"{model_dir}/Qwen2.5-Math-1.5B"}
config.mode = "mean"
config.langs = ["en"]
config.eval_batch_size = 8

# ========== Estimation Parameters ==========
est_parameters = ConfigPE(
    thetas=[None, None],
    weights=anchors_weights,
    sample_ids=anchors,
    bench=config.bench,
    mode=config.mode,
    correct_metric=config.metric,
)

# ========== Merger ==========
path_to_store_yaml = f"{config.path_to_store_config}/{config.run_id}"
merger = TaskArithmeticMerger(
    run_id=config.run_id,
    path_to_base_model=config.base_model,
    model_paths=[config.models[lang_id]],
    path_to_store_yaml=path_to_store_yaml,
    path_to_store_merged_model=config.path_to_store_merged_model,
    dtype=config.dtype,
)

# ========== Problem ==========
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
problem = MathReasoningProblem(
    merger=merger,
    test_df=None,
    search_df=None,
    lm_eval_tasks=config.tasks,
    lang_id=lang_id,
    conf_pe=est_parameters,
    device=config.device,
    n_var=1, 
    n_obj=2,
    n_eq_constr=0,
    n_ieq_constr=0,
    discrete=True,
    eval_batch_size=config.eval_batch_size,
    additional_templates_folder=config.additional_templates_folder,
    tokenizer=tokenizer
)

# ========== Algorithm and Search ==========
config.pop_size = 20
config.n_iter = 10
run_id = config.run_id

algorithm = NSGA2(
    pop_size=config.pop_size,
    sampling=IntegerRandomSampling(),    
    crossover=SBX(),
    mutation=PM(),
    eliminate_duplicates=True,
)

results_path = f"{config.path_to_store_config}/{run_id}/"

searcher = Searcher(
    problem=problem,
    n_iter=config.n_iter,
    algorithm=algorithm,
    results_path=results_path,
    run_id=config.run_id,
    seed=config.seed,
    verbose=False,
)

result_df = searcher.search()
print("✅ Evolutionary search completed.")

#searcher.test()
#print("✅ Final model test done.")
#print("📁 Results saved in:", results_path)