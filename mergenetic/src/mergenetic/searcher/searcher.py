from logging import getLogger
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.algorithm import Algorithm
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from mergenetic.optimization import MergingProblem

logger = getLogger(__name__)

class Searcher:
    def __init__(
        self,
        problem: MergingProblem,
        algorithm: Algorithm,
        results_path: str,
        n_iter: int,
        seed: int,
        run_id: str,
        verbose: bool = True,
    ):
        self.problem = problem
        self.algorithm = algorithm
        self.results_path = Path(results_path)
        self.n_iter = n_iter
        self.seed = seed
        self.run_id = run_id
        self.verbose = verbose

    def search(self) -> pd.DataFrame | None:
        termination = get_termination("n_gen", self.n_iter)
        
        result = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination = termination,
            seed=self.seed,
            verbose=self.verbose,
        )
        
        self.result_X = result.X / 10 if self.problem.discrete else result.X 
        self.result_F = result.F       
        
        # === Save Parento Front solutions in a separate CSV (objectives first, followed by genotypes) ===
        F_df = pd.DataFrame(self.result_F, columns=[f"f{i}" for i in range(self.result_F.shape[1])])
        pareto_df = pd.DataFrame(self.result_X, columns=[f"x{i}" for i in range(self.result_X.shape[1])])
        combined = pd.concat([F_df, pareto_df], axis=1)

        pareto_csv_path = self.results_path / f"{self.run_id}_solutions.csv"
        combined.to_csv(pareto_csv_path, index=False)

        logger.info(f"Pareto front saved in CSV: {pareto_csv_path}")                           
        logger.info(f"Best solution found: {self.result_X}. Objective value: {self.result_F}")

        if not hasattr(self.problem, "results_df"):
            logger.warning("No results_df attribute found in problem.")
            return None

        if isinstance(self.problem.results_df, pd.DataFrame):
            self.problem.results_df.to_csv(self.results_path / f"{self.run_id}.csv", index=False)
            return self.problem.results_df

        elif isinstance(self.problem.results_df, dict):
            for key, df in self.problem.results_df.items():
                df.to_csv(self.results_path / f"{self.run_id}_{key}.csv", index=False)
            return self.problem.results_df

        logger.error("problem.results_df must be a DataFrame or a dict of DataFrames.")
        return None


    def test(self):
        """
        Evaluate best solution(s) on the test set and save fitness + genotypes only.
        """
        logger.info("Starting final evaluation on test set.")

        if not hasattr(self, "result_X") or not hasattr(self, "result_F"):
            logger.error("Missing result_X or result_F. You must run search() before test().")
            return

        # Ensure arrays are 2D
        X = self.result_X if self.result_X.ndim > 1 else np.array([self.result_X])
        F = self.result_F if self.result_F.ndim > 1 else np.array([self.result_F])

        test_records = []

        for i, genotype in enumerate(X):
            assert isinstance(genotype, np.ndarray), "Genotype is not a NumPy array."

            fitness, _ = self.problem.test(genotype)
            logger.info(f"[{i+1}] Genotype {genotype} got fitness {fitness}")

            if not isinstance(fitness, (list, np.ndarray)):
                fitness = [fitness]

            row = {f"objective_{i+1}": val for i, val in enumerate(fitness)}
            row.update({f"genotype_{i+1}": val for i, val in enumerate(genotype.tolist())})

            test_records.append(row)

        df = pd.DataFrame(test_records)
        df.to_csv(self.results_path / f"{self.run_id}_test.csv", index=False)

        logger.info("Test evaluation completed.")
    

    def visualize_results(self) -> None:
        """Plot optimization metrics and phenotypes from results_df."""
        if not hasattr(self.problem, "results_df"):
            raise AttributeError("Problem does not have 'results_df' to visualize.")

        df = self.problem.results_df

        if isinstance(df, dict):
            for key, sub_df in df.items():
                self._plot_metrics(sub_df, title_prefix=key)
        else:
            self._plot_metrics(df)

    def _plot_metrics(self, df: pd.DataFrame, title_prefix: str = ""):
        metrics = [col for col in df.columns if "objective" in col]
        phenotypes = [col for col in df.columns if "phenotype" in col]

        for metric in metrics:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[metric], marker="o")
            plt.title(f" Metric: {title_prefix}")
            plt.xlabel("Step")
            plt.ylabel(metric)
            plt.grid(True)
            plt.show()

        for phenotype in phenotypes:
            plt.figure(figsize=(10, 4))
            plt.plot(df["step"], df[phenotype], marker="x", linestyle="--")
            plt.title(f"{title_prefix} Phenotype: {phenotype}")
            plt.xlabel("Step")
            plt.ylabel(phenotype)
            plt.grid(True)
            plt.show()
            
            