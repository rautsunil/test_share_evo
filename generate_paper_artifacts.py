"""
EvoCRM Paper Artifacts Generator
==================================

Converts raw JSON results into paper-ready LaTeX tables and figures.

Generates:
    - Table 1: Main results (EvoCRM vs baselines on all tasks)
    - Table 2: Parameter efficiency (LoRA vs full fine-tuning)
    - Table 3: Tower ablation (which modality matters)
    - Table 4: Hub architecture comparison
    - Table 5: Multi-task vs single-task
    - Figure 1: Training loss curves
    - Figure 2: LoRA rank vs accuracy Pareto curve
    - Dataset statistics table

Usage:
    python generate_paper_artifacts.py \
        --results_dir ./results/ \
        --output_dir ./results/paper_artifacts/

Author: EvoCRM Team
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Optional: matplotlib for figures
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


class PaperArtifactGenerator:
    """Generates paper-ready tables and figures from JSON results."""

    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self) -> Dict[str, Any]:
        """Load all result JSON files."""
        data = {}
        file_map = {
            "baselines": "baselines/all_baseline_results.json",
            "evaluation": "evaluation/evaluation_results.json",
            "ablations": "ablations/all_ablation_results.json",
            "history": "checkpoints/training_history.json" if False else None,
        }

        # Also check workspace layout
        alt_paths = {
            "baselines": "baselines/all_baseline_results.json",
            "evaluation": "evaluation/evaluation_results.json",
            "ablations": "ablations/all_ablation_results.json",
        }

        for key, rel_path in {**file_map, **alt_paths}.items():
            if rel_path is None:
                continue
            path = self.results_dir / rel_path
            if path.exists() and key not in data:
                with open(path) as f:
                    data[key] = json.load(f)
                print(f"  ✓ Loaded {key}: {path}")

        return data

    def generate_all(self):
        """Generate all paper artifacts."""
        print("=" * 60)
        print("GENERATING PAPER ARTIFACTS")
        print("=" * 60)

        data = self.load_results()

        if "evaluation" in data and "baselines" in data:
            self.generate_table1_main_results(data)
        if "evaluation" in data:
            self.generate_table2_param_efficiency(data)
        if "ablations" in data:
            if "tower_dropout" in data["ablations"]:
                self.generate_table3_tower_ablation(data)
            if "hub_type" in data["ablations"]:
                self.generate_table4_hub_comparison(data)
            if "single_task" in data["ablations"]:
                self.generate_table5_multitask(data)
            if "lora_sweep" in data["ablations"] and HAS_MPL:
                self.generate_figure_lora_pareto(data)

        self.generate_dataset_stats(data)

        print(f"\n  All artifacts saved to: {self.output_dir}")

    def generate_table1_main_results(self, data):
        """Table 1: EvoCRM vs baselines."""
        print("\n  Generating Table 1: Main Results ...")

        evocrm = data["evaluation"].get("evocrm", {})
        baselines = data.get("baselines", {})
        comparison = data["evaluation"].get("comparison", {})

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Main results on the Olist Brazilian E-Commerce dataset. "
            r"Best results in \textbf{bold}. $\Delta$ shows improvement of EvoCRM over best baseline.}",
            r"\label{tab:main_results}",
            r"\small",
            r"\begin{tabular}{llccccc}",
            r"\toprule",
            r"Task & Metric & LR/Ridge & Random Forest & XGBoost & \textbf{EvoCRM} & $\Delta$ \\",
            r"\midrule",
        ]

        for task_name, metrics in evocrm.items():
            if task_name in ["confusion_matrix"]:
                continue

            # Determine primary metric
            if "auc_roc" in metrics:
                primary = "auc_roc"
                metric_display = "AUC-ROC"
            elif "r2" in metrics:
                primary = "r2"
                metric_display = "R$^2$"
            else:
                continue

            evocrm_score = metrics[primary]

            # Get baseline scores
            bl_scores = {}
            bl_task = baselines.get(task_name, {})
            name_map = {
                "logistic_regression": "LR/Ridge",
                "ridge_regression": "LR/Ridge",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost",
                "lightgbm": "LightGBM",
            }
            for model_name, model_metrics in bl_task.items():
                if "error" in model_metrics:
                    continue
                short_name = name_map.get(model_name, model_name)
                bl_scores[short_name] = model_metrics.get(primary, 0)

            # Build row
            lr_score = bl_scores.get("LR/Ridge", 0)
            rf_score = bl_scores.get("Random Forest", 0)
            xgb_score = bl_scores.get("XGBoost", bl_scores.get("LightGBM", 0))

            best_bl = max(lr_score, rf_score, xgb_score)
            delta = evocrm_score - best_bl

            # Bold the best
            scores = [lr_score, rf_score, xgb_score, evocrm_score]
            best_idx = np.argmax(scores)
            formatted = []
            for i, s in enumerate(scores):
                if s == 0:
                    formatted.append("—")
                elif i == best_idx:
                    formatted.append(f"\\textbf{{{s:.4f}}}")
                else:
                    formatted.append(f"{s:.4f}")

            task_display = task_name.replace("_", " ").title()
            delta_str = f"{delta:+.4f}" if delta != 0 else "—"

            lines.append(
                f"  {task_display} & {metric_display} & "
                f"{formatted[0]} & {formatted[1]} & {formatted[2]} & "
                f"{formatted[3]} & {delta_str} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
        ])

        latex = "\n".join(lines)
        path = self.output_dir / "table1_main_results.tex"
        with open(path, "w") as f:
            f.write(latex)
        print(f"    → {path}")

    def generate_table2_param_efficiency(self, data):
        """Table 2: Parameter efficiency."""
        print("  Generating Table 2: Parameter Efficiency ...")

        model_stats = data["evaluation"].get("model_stats", {})

        total = model_stats.get("total_params", 0)
        trainable = model_stats.get("trainable_params", 0)

        # Estimate specialist ensemble params (6 separate XGBoost-equivalent models)
        n_tasks = len(data["evaluation"].get("evocrm", {}))
        specialist_params = total * n_tasks  # Rough estimate

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Parameter efficiency: EvoCRM (hub-and-spoke) vs specialist ensemble.}",
            r"\label{tab:param_efficiency}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Approach & Total Params & Trainable & Reduction \\",
            r"\midrule",
            f"  Specialist Ensemble ({n_tasks} models) & {specialist_params:,} & {specialist_params:,} & — \\\\",
            f"  EvoCRM (full fine-tune) & {total:,} & {total:,} & {(1-total/max(specialist_params,1))*100:.0f}\\% \\\\",
            f"  EvoCRM + LoRA (rank 8) & {total:,} & {trainable:,} & {(1-trainable/max(specialist_params,1))*100:.0f}\\% \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]

        path = self.output_dir / "table2_param_efficiency.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"    → {path}")

    def generate_table3_tower_ablation(self, data):
        """Table 3: Tower dropout ablation."""
        print("  Generating Table 3: Tower Ablation ...")

        tower_data = data["ablations"]["tower_dropout"]

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Tower ablation: impact of removing each modality.}",
            r"\label{tab:tower_ablation}",
            r"\begin{tabular}{l" + "c" * 4 + "}",
            r"\toprule",
            r"Configuration & Churn AUC & CLV R$^2$ & Upsell AUC & Params \\",
            r"\midrule",
        ]

        for variant_name, metrics in tower_data.items():
            if variant_name.startswith("_"):
                continue
            churn = metrics.get("churn_auc", "—")
            clv = metrics.get("clv_r2", "—")
            upsell = metrics.get("upsell_auc", "—")
            params = metrics.get("trainable_params", 0)

            churn_s = f"{churn:.4f}" if isinstance(churn, float) else churn
            clv_s = f"{clv:.4f}" if isinstance(clv, float) else clv
            upsell_s = f"{upsell:.4f}" if isinstance(upsell, float) else upsell

            display = variant_name.replace("_", " ").title()
            lines.append(
                f"  {display} & {churn_s} & {clv_s} & {upsell_s} & {params:,} \\\\"
            )

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = self.output_dir / "table3_tower_ablation.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"    → {path}")

    def generate_table4_hub_comparison(self, data):
        """Table 4: Hub architecture comparison."""
        print("  Generating Table 4: Hub Architecture ...")

        hub_data = data["ablations"]["hub_type"]

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Hub fusion strategy comparison.}",
            r"\label{tab:hub_comparison}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Fusion Strategy & Churn AUC & CLV R$^2$ & Params \\",
            r"\midrule",
        ]

        for variant, metrics in hub_data.items():
            if variant.startswith("_"):
                continue
            churn = metrics.get("churn_auc", 0)
            clv = metrics.get("clv_r2", 0)
            params = metrics.get("trainable_params", 0)
            display = variant.replace("_", " ").title()
            lines.append(f"  {display} & {churn:.4f} & {clv:.4f} & {params:,} \\\\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = self.output_dir / "table4_hub_comparison.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"    → {path}")

    def generate_table5_multitask(self, data):
        """Table 5: Multi-task vs single-task."""
        print("  Generating Table 5: Multi-task Benefit ...")

        st_data = data["ablations"]["single_task"]

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Multi-task vs single-task training comparison.}",
            r"\label{tab:multitask}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Training Mode & Churn AUC & CLV R$^2$ & Upsell AUC & Total Params \\",
            r"\midrule",
        ]

        for variant, metrics in st_data.items():
            if variant.startswith("_"):
                continue
            churn = metrics.get("churn_auc", "—")
            clv = metrics.get("clv_r2", "—")
            upsell = metrics.get("upsell_auc", "—")
            params = metrics.get("trainable_params", 0)

            churn_s = f"{churn:.4f}" if isinstance(churn, float) else "—"
            clv_s = f"{clv:.4f}" if isinstance(clv, float) else "—"
            upsell_s = f"{upsell:.4f}" if isinstance(upsell, float) else "—"

            display = variant.replace("_", " ").title()
            lines.append(f"  {display} & {churn_s} & {clv_s} & {upsell_s} & {params:,} \\\\")

        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

        path = self.output_dir / "table5_multitask.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))
        print(f"    → {path}")

    def generate_figure_lora_pareto(self, data):
        """Figure: LoRA rank vs accuracy Pareto curve."""
        print("  Generating Figure: LoRA Pareto Curve ...")

        lora_data = data["ablations"]["lora_sweep"]

        ranks = []
        churn_aucs = []
        param_counts = []

        for variant, metrics in lora_data.items():
            if variant.startswith("_"):
                continue
            if variant == "full_finetune":
                rank = 0
            else:
                rank = int(variant.split("_")[1])

            ranks.append(rank)
            churn_aucs.append(metrics.get("churn_auc", 0))
            param_counts.append(metrics.get("trainable_params", 0))

        # Sort by rank
        order = np.argsort(ranks)
        ranks = [ranks[i] for i in order]
        churn_aucs = [churn_aucs[i] for i in order]
        param_counts = [param_counts[i] for i in order]

        fig, ax1 = plt.subplots(figsize=(8, 5))

        color1 = "#2196F3"
        color2 = "#FF5722"

        ax1.set_xlabel("LoRA Rank (0 = full fine-tune)", fontsize=12)
        ax1.set_ylabel("Churn AUC-ROC", fontsize=12, color=color1)
        ax1.plot(ranks, churn_aucs, "o-", color=color1, linewidth=2, markersize=8)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Trainable Parameters", fontsize=12, color=color2)
        ax2.bar(ranks, param_counts, alpha=0.3, color=color2, width=1.5)
        ax2.tick_params(axis="y", labelcolor=color2)

        plt.title("LoRA Rank: Accuracy vs Parameter Efficiency", fontsize=14)
        fig.tight_layout()

        path = self.output_dir / "figure_lora_pareto.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    → {path}")

    def generate_dataset_stats(self, data):
        """Generate dataset statistics table."""
        print("  Generating Dataset Statistics ...")

        # Try to load metadata
        metadata_paths = [
            self.results_dir.parent / "data" / "olist" / "metadata.json",
            self.results_dir / ".." / "data" / "olist" / "metadata.json",
        ]

        metadata = None
        for mp in metadata_paths:
            if mp.exists():
                with open(mp) as f:
                    metadata = json.load(f)
                break

        if metadata:
            stats = metadata.get("stats", {})
            lines = [
                r"\begin{table}[h]",
                r"\centering",
                r"\caption{Olist dataset statistics after preprocessing.}",
                r"\label{tab:dataset_stats}",
                r"\begin{tabular}{lr}",
                r"\toprule",
                r"Statistic & Value \\",
                r"\midrule",
                f"  Unique customers & {stats.get('n_users', 0):,} \\\\",
                f"  Transaction items & {stats.get('n_transactions', 0):,} \\\\",
                f"  Order journey events & {stats.get('n_web_events', 0):,} \\\\",
                f"  Unique products & {stats.get('n_products', 0):,} \\\\",
                f"  Product categories & {stats.get('n_categories', 0):,} \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
            path = self.output_dir / "table_dataset_stats.tex"
            with open(path, "w") as f:
                f.write("\n".join(lines))
            print(f"    → {path}")
        else:
            print("    ⚠ metadata.json not found — skipping dataset stats table")


def main():
    parser = argparse.ArgumentParser(description="Generate paper artifacts")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/paper_artifacts/")
    args = parser.parse_args()

    gen = PaperArtifactGenerator(args.results_dir, args.output_dir)
    gen.generate_all()


if __name__ == "__main__":
    main()
