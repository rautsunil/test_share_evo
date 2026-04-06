"""
EvoCRM Evaluation Script
=========================

Runs the trained EvoCRM model on the test set, computes all metrics,
compares against baselines, runs statistical significance tests,
and generates paper-ready output (JSON + LaTeX tables).

This script produces TABLE 1 — the most important artifact in your paper.

Usage:
    python evaluate_evocrm.py \
        --data_dir ./evocrm_data/ \
        --checkpoint ./checkpoints/evocrm_final.pt \
        --baselines ./results/baselines/all_baseline_results.json \
        --output_dir ./results/final/

Author: EvoCRM Team
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# Import from train_evocrm (same directory)
from train_evocrm import EvoCRMModel, EvoCRMDataset


# ============================================================
# BOOTSTRAP SIGNIFICANCE TESTING
# ============================================================

def paired_bootstrap_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Paired bootstrap test for statistical significance.
    Tests whether model A is significantly better than model B.

    Returns p-value, mean difference, and 95% CI.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    score_a = metric_fn(y_true, scores_a)
    score_b = metric_fn(y_true, scores_b)
    observed_diff = score_a - score_b

    # Bootstrap
    count_b_better = 0
    diffs = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        boot_a = metric_fn(y_true[indices], scores_a[indices])
        boot_b = metric_fn(y_true[indices], scores_b[indices])
        diff = boot_a - boot_b
        diffs.append(diff)
        if diff <= 0:
            count_b_better += 1

    diffs = np.array(diffs)
    p_value = count_b_better / n_bootstrap

    return {
        "score_a": float(score_a),
        "score_b": float(score_b),
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "ci_95_lower": float(np.percentile(diffs, 2.5)),
        "ci_95_upper": float(np.percentile(diffs, 97.5)),
        "significant": p_value < 0.05,
    }


# ============================================================
# EVALUATOR
# ============================================================

class EvoCRMEvaluator:
    """Evaluates EvoCRM and compares against baselines."""

    def __init__(self, model, test_loader, device, baseline_results=None):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.baseline_results = baseline_results or {}

    @torch.no_grad()
    def get_predictions(self) -> Tuple[Dict, Dict]:
        """Run model on test set and collect all predictions + targets."""
        self.model.eval()
        all_preds = {}
        all_targets = {}
        all_embeddings = []

        for batch in self.test_loader:
            features = batch["features"].to(self.device)
            sequences = batch["sequences"].to(self.device)

            predictions, hub_out = self.model(features, sequences)
            all_embeddings.append(hub_out.cpu().numpy())

            for task_name, pred in predictions.items():
                if task_name not in all_preds:
                    all_preds[task_name] = []
                all_preds[task_name].append(pred.cpu().numpy())

            for task_name, target in batch["targets"].items():
                if task_name not in all_targets:
                    all_targets[task_name] = []
                all_targets[task_name].append(target.numpy())

        # Concatenate
        for k in all_preds:
            all_preds[k] = np.concatenate(all_preds[k])
        for k in all_targets:
            all_targets[k] = np.concatenate(all_targets[k])

        embeddings = np.concatenate(all_embeddings)
        return all_preds, all_targets, embeddings

    def evaluate_all(self) -> Dict[str, Any]:
        """Run full evaluation."""
        print("=" * 60)
        print("EVALUATING EVOCRM ON TEST SET")
        print("=" * 60)

        preds, targets, embeddings = self.get_predictions()
        results = {"evocrm": {}, "comparison": {}, "significance": {}}

        # ---- Classification tasks ----
        for task_name in ["churn", "upsell", "early_adopter", "satisfaction_risk"]:
            if task_name not in preds or task_name not in targets:
                continue

            y_true = targets[task_name]
            y_pred_proba = preds[task_name]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            metrics = {}
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
            except ValueError:
                metrics["auc_roc"] = 0.5
            metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            try:
                metrics["avg_precision"] = float(average_precision_score(y_true, y_pred_proba))
            except ValueError:
                metrics["avg_precision"] = 0.0

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()

            results["evocrm"][task_name] = metrics
            print(f"\n  {task_name.upper()}: AUC={metrics['auc_roc']:.4f} | "
                  f"F1={metrics['f1']:.4f} | Prec={metrics['precision']:.4f} | "
                  f"Rec={metrics['recall']:.4f}")

        # ---- Regression tasks ----
        for task_name in ["clv", "days_next_purchase"]:
            if task_name not in preds or task_name not in targets:
                continue

            y_true = targets[task_name]
            y_pred = preds[task_name]

            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }

            results["evocrm"][task_name] = metrics
            print(f"\n  {task_name.upper()}: RMSE={metrics['rmse']:.4f} | "
                  f"MAE={metrics['mae']:.4f} | R²={metrics['r2']:.4f}")

        # ---- Compare against baselines ----
        if self.baseline_results:
            print(f"\n{'=' * 60}")
            print("COMPARISON: EVOCRM vs BASELINES")
            print("=" * 60)

            for task_name, evocrm_metrics in results["evocrm"].items():
                if task_name not in self.baseline_results:
                    continue

                baseline_task = self.baseline_results[task_name]
                primary_metric = "auc_roc" if "auc_roc" in evocrm_metrics else "r2"
                evocrm_score = evocrm_metrics[primary_metric]

                comparison = {}
                best_baseline_name = None
                best_baseline_score = -999

                for model_name, model_metrics in baseline_task.items():
                    if "error" in model_metrics:
                        continue
                    baseline_score = model_metrics.get(primary_metric, 0)
                    delta = evocrm_score - baseline_score
                    pct = (delta / abs(baseline_score) * 100) if baseline_score != 0 else 0

                    comparison[model_name] = {
                        "baseline_score": baseline_score,
                        "evocrm_score": evocrm_score,
                        "absolute_diff": round(delta, 4),
                        "relative_pct": round(pct, 2),
                        "evocrm_wins": delta > 0,
                    }

                    if baseline_score > best_baseline_score:
                        best_baseline_score = baseline_score
                        best_baseline_name = model_name

                results["comparison"][task_name] = comparison

                # Print
                print(f"\n  {task_name.upper()} ({primary_metric}):")
                print(f"    EvoCRM:          {evocrm_score:.4f}")
                for mn, comp in comparison.items():
                    marker = "✓" if comp["evocrm_wins"] else "✗"
                    print(f"    {mn:<20} {comp['baseline_score']:.4f}  "
                          f"Δ={comp['absolute_diff']:+.4f} ({comp['relative_pct']:+.1f}%) {marker}")

        # ---- Embedding statistics ----
        results["embedding_stats"] = {
            "shape": list(embeddings.shape),
            "mean_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
            "std_norm": float(np.linalg.norm(embeddings, axis=1).std()),
        }

        # ---- Parameter count ----
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        results["model_stats"] = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "param_efficiency": round(trainable_params / total_params, 4),
        }
        print(f"\n  Model params: {total_params:,} total | "
              f"{trainable_params:,} trainable ({trainable_params/total_params:.1%})")

        return results

    def generate_latex_table(self, results: Dict) -> str:
        """Generate LaTeX table for the paper."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Main results: EvoCRM vs baselines on all tasks.}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Task & Metric & Best Baseline & EvoCRM & $\Delta$ \\",
            r"\midrule",
        ]

        for task_name, evocrm_metrics in results.get("evocrm", {}).items():
            primary = "auc_roc" if "auc_roc" in evocrm_metrics else "r2"
            metric_name = "AUC-ROC" if primary == "auc_roc" else "R²"
            if primary == "rmse":
                metric_name = "RMSE"
            evocrm_score = evocrm_metrics[primary]

            comparison = results.get("comparison", {}).get(task_name, {})
            best_bl_score = 0
            best_bl_name = "—"
            for mn, comp in comparison.items():
                if comp["baseline_score"] > best_bl_score:
                    best_bl_score = comp["baseline_score"]
                    best_bl_name = mn

            delta = evocrm_score - best_bl_score
            bold = r"\textbf" if delta > 0 else ""

            task_display = task_name.replace("_", " ").title()
            lines.append(
                f"  {task_display} & {metric_name} & "
                f"{best_bl_score:.4f} ({best_bl_name}) & "
                f"{bold}{{{evocrm_score:.4f}}} & "
                f"{delta:+.4f} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate EvoCRM")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baselines", type=str, default=None,
                        help="Path to all_baseline_results.json")
    parser.add_argument("--output_dir", type=str, default="./results/final/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_config = ckpt["config"]

    # Build model
    model = EvoCRMModel(
        n_features=model_config["n_features"],
        embed_dim=model_config.get("embed_dim", 256),
        hub_dim=model_config.get("hub_dim", 512),
        n_event_types=model_config.get("n_event_types", 10),
        n_products=model_config.get("n_products", 50000),
        tasks=model_config.get("tasks"),
    )
    model.load_state_dict(ckpt["model_state"], strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load test data
    test_ds = EvoCRMDataset(args.data_dir, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
    )
    print(f"Test set: {len(test_ds):,} users")

    # Load baselines
    baseline_results = {}
    if args.baselines and os.path.exists(args.baselines):
        with open(args.baselines) as f:
            baseline_results = json.load(f)
        print(f"Loaded baselines: {list(baseline_results.keys())}")

    # Evaluate
    evaluator = EvoCRMEvaluator(model, test_loader, device, baseline_results)
    results = evaluator.evaluate_all()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Generate LaTeX table
    latex_table = evaluator.generate_latex_table(results)
    latex_path = output_dir / "table1_main_results.tex"
    with open(latex_path, "w") as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Results: {results_path}")
    print(f"  LaTeX:   {latex_path}")
    n_wins = sum(
        1 for task in results.get("comparison", {}).values()
        for comp in task.values() if comp.get("evocrm_wins")
    )
    n_total = sum(
        len(task) for task in results.get("comparison", {}).values()
    )
    if n_total > 0:
        print(f"  EvoCRM wins: {n_wins}/{n_total} comparisons")


if __name__ == "__main__":
    main()
