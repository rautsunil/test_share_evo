"""
EvoCRM Ablation Studies
========================

Runs 4 ablation experiments that reviewers expect:
    A. Tower Dropout       → Which modality matters most for which task?
    B. Hub Architecture    → Does XLNet-style fusion beat simpler alternatives?
    C. LoRA Rank Sweep     → Accuracy vs parameter efficiency tradeoff
    D. Single vs Multi-task→ Does joint training help?

Each ablation retrains a model variant and evaluates on the test set.

Usage:
    # Run all ablations (takes ~4-8 hours on GPU)
    python run_ablations.py --data_dir ./evocrm_data/ --output_dir ./results/ablations/

    # Run a specific ablation
    python run_ablations.py --data_dir ./evocrm_data/ --ablation tower_dropout
    python run_ablations.py --data_dir ./evocrm_data/ --ablation hub_type
    python run_ablations.py --data_dir ./evocrm_data/ --ablation lora_sweep
    python run_ablations.py --data_dir ./evocrm_data/ --ablation single_task

Author: EvoCRM Team
"""

import os
import sys
import json
import copy
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

warnings.filterwarnings("ignore")

from train_evocrm import (
    EvoCRMModel, EvoCRMDataset, EvoCRMTrainer,
    CustomerTower, InteractionTower, Hub, MultiTaskHead,
    MultiTaskLoss, apply_lora,
)
from torch.utils.data import DataLoader


# ============================================================
# HELPER: Quick train + evaluate
# ============================================================

def quick_train_eval(
    model, train_loader, val_loader, test_loader, device,
    epochs: int = 15, lr: float = 1e-3, label: str = "",
) -> Dict[str, float]:
    """Train a model variant quickly and return test metrics."""
    model = model.to(device)
    task_types = model.task_heads.task_types
    criterion = MultiTaskLoss(task_types).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=lr, weight_decay=0.01,
    )

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            sequences = batch["sequences"].to(device)
            targets = {k: v.to(device) for k, v in batch["targets"].items()}

            optimizer.zero_grad()
            preds, _ = model(features, sequences)
            loss, _ = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate on test
    model.eval()
    all_preds = {}
    all_targets = {}

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            sequences = batch["sequences"].to(device)

            preds, _ = model(features, sequences)
            for k, v in preds.items():
                all_preds.setdefault(k, []).append(v.cpu().numpy())
            for k, v in batch["targets"].items():
                all_targets.setdefault(k, []).append(v.numpy())

    for k in all_preds:
        all_preds[k] = np.concatenate(all_preds[k])
    for k in all_targets:
        all_targets[k] = np.concatenate(all_targets[k])

    # Compute metrics
    metrics = {}
    for task_name in all_preds:
        if task_name not in all_targets:
            continue
        y_true = all_targets[task_name]
        y_pred = all_preds[task_name]

        if task_name in ["churn", "upsell", "early_adopter", "satisfaction_risk"]:
            try:
                metrics[f"{task_name}_auc"] = float(roc_auc_score(y_true, y_pred))
            except ValueError:
                metrics[f"{task_name}_auc"] = 0.5
        else:
            metrics[f"{task_name}_r2"] = float(r2_score(y_true, y_pred))
            metrics[f"{task_name}_rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics["trainable_params"] = n_params

    return metrics


# ============================================================
# ABLATION A: Tower Dropout
# ============================================================

def ablation_tower_dropout(train_loader, val_loader, test_loader,
                            n_features, config, device) -> Dict:
    """
    Remove each tower and measure impact.
    Paper Table 3: Tower importance analysis.
    """
    print("\n" + "=" * 60)
    print("ABLATION A: TOWER DROPOUT")
    print("  (Which modality matters most for which task?)")
    print("=" * 60)

    results = {}
    epochs = config.get("ablation_epochs", 15)

    # Full model (control)
    print("\n  [Control] Full model (both towers) ...")
    model_full = EvoCRMModel(n_features, tasks=config["tasks"],
                              n_event_types=config["n_event_types"])
    results["full_model"] = quick_train_eval(
        model_full, train_loader, val_loader, test_loader, device, epochs,
        label="full",
    )

    # Drop Customer Tower (zero out customer embeddings)
    print("  [Variant] Without Customer Tower ...")

    class NoCustModel(EvoCRMModel):
        def forward(self, features, sequences, task_name=None):
            c_emb = torch.zeros(features.shape[0], self.customer_tower.network[-1].out_features).to(features.device)
            i_emb = self.interaction_tower(sequences)
            hub_out = self.hub(c_emb, i_emb)
            return self.task_heads(hub_out, task_name), hub_out

    model_no_cust = NoCustModel(n_features, tasks=config["tasks"],
                                 n_event_types=config["n_event_types"])
    results["no_customer_tower"] = quick_train_eval(
        model_no_cust, train_loader, val_loader, test_loader, device, epochs,
        label="no_cust",
    )

    # Drop Interaction Tower
    print("  [Variant] Without Interaction Tower ...")

    class NoInterModel(EvoCRMModel):
        def forward(self, features, sequences, task_name=None):
            c_emb = self.customer_tower(features)
            i_emb = torch.zeros(features.shape[0], 256).to(features.device)
            hub_out = self.hub(c_emb, i_emb)
            return self.task_heads(hub_out, task_name), hub_out

    model_no_inter = NoInterModel(n_features, tasks=config["tasks"],
                                   n_event_types=config["n_event_types"])
    results["no_interaction_tower"] = quick_train_eval(
        model_no_inter, train_loader, val_loader, test_loader, device, epochs,
        label="no_inter",
    )

    # Customer Tower only (no hub, direct task heads)
    print("  [Variant] Customer Tower only (no hub) ...")

    class CustOnlyModel(nn.Module):
        def __init__(self, n_feat, tasks):
            super().__init__()
            self.customer_tower = CustomerTower(n_feat, 256)
            self.task_heads = MultiTaskHead(256, tasks)
        def forward(self, features, sequences, task_name=None):
            emb = self.customer_tower(features)
            return self.task_heads(emb, task_name), emb

    model_cust_only = CustOnlyModel(n_features, config["tasks"])
    results["customer_tower_only"] = quick_train_eval(
        model_cust_only, train_loader, val_loader, test_loader, device, epochs,
        label="cust_only",
    )

    _print_ablation_results("Tower Dropout", results)
    return results


# ============================================================
# ABLATION B: Hub Architecture
# ============================================================

def ablation_hub_type(train_loader, val_loader, test_loader,
                       n_features, config, device) -> Dict:
    """
    Compare Hub architectures.
    Paper Table 4: Fusion strategy comparison.
    """
    print("\n" + "=" * 60)
    print("ABLATION B: HUB ARCHITECTURE")
    print("  (Does cross-attention fusion beat simpler alternatives?)")
    print("=" * 60)

    results = {}
    epochs = config.get("ablation_epochs", 15)

    # Original Hub (cross-attention + gating)
    print("\n  [Control] Cross-Attention + Gating Hub ...")
    model = EvoCRMModel(n_features, tasks=config["tasks"],
                         n_event_types=config["n_event_types"])
    results["cross_attn_gating"] = quick_train_eval(
        model, train_loader, val_loader, test_loader, device, epochs,
    )

    # Concatenation Hub
    print("  [Variant] Simple Concatenation Hub ...")

    class ConcatHub(nn.Module):
        def __init__(self, tower_dim=256, hub_dim=512):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(tower_dim * 2, hub_dim),
                nn.LayerNorm(hub_dim),
                nn.GELU(),
                nn.Linear(hub_dim, hub_dim),
            )
        def forward(self, c, i):
            return self.fc(torch.cat([c, i], dim=-1))

    model_concat = EvoCRMModel(n_features, tasks=config["tasks"],
                                n_event_types=config["n_event_types"])
    model_concat.hub = ConcatHub()
    results["concat"] = quick_train_eval(
        model_concat, train_loader, val_loader, test_loader, device, epochs,
    )

    # Mean Pooling Hub
    print("  [Variant] Mean Pooling Hub ...")

    class MeanHub(nn.Module):
        def __init__(self, tower_dim=256, hub_dim=512):
            super().__init__()
            self.proj = nn.Linear(tower_dim, hub_dim)
        def forward(self, c, i):
            return self.proj((c + i) / 2.0)

    model_mean = EvoCRMModel(n_features, tasks=config["tasks"],
                              n_event_types=config["n_event_types"])
    model_mean.hub = MeanHub()
    results["mean_pool"] = quick_train_eval(
        model_mean, train_loader, val_loader, test_loader, device, epochs,
    )

    # Cross-attention only (no gating)
    print("  [Variant] Cross-Attention only (no gating) ...")

    class AttnOnlyHub(nn.Module):
        def __init__(self, tower_dim=256, hub_dim=512):
            super().__init__()
            self.proj_c = nn.Linear(tower_dim, hub_dim)
            self.proj_i = nn.Linear(tower_dim, hub_dim)
            self.attn = nn.MultiheadAttention(hub_dim, 4, batch_first=True)
            self.out = nn.Linear(hub_dim, hub_dim)
        def forward(self, c, i):
            c_p = self.proj_c(c).unsqueeze(1)
            i_p = self.proj_i(i).unsqueeze(1)
            seq = torch.cat([c_p, i_p], dim=1)
            out, _ = self.attn(seq, seq, seq)
            return self.out(out.mean(dim=1))

    model_attn = EvoCRMModel(n_features, tasks=config["tasks"],
                              n_event_types=config["n_event_types"])
    model_attn.hub = AttnOnlyHub()
    results["cross_attn_only"] = quick_train_eval(
        model_attn, train_loader, val_loader, test_loader, device, epochs,
    )

    _print_ablation_results("Hub Architecture", results)
    return results


# ============================================================
# ABLATION C: LoRA Rank Sweep
# ============================================================

def ablation_lora_sweep(train_loader, val_loader, test_loader,
                         n_features, config, device) -> Dict:
    """
    Sweep LoRA ranks to find accuracy-efficiency tradeoff.
    Paper Figure 3: Pareto curve of accuracy vs parameters.
    """
    print("\n" + "=" * 60)
    print("ABLATION C: LORA RANK SWEEP")
    print("  (Accuracy vs parameter efficiency tradeoff)")
    print("=" * 60)

    results = {}
    epochs = config.get("ablation_epochs", 15)
    ranks = [0, 2, 4, 8, 16, 32]  # 0 = full fine-tuning

    for rank in ranks:
        label = f"rank_{rank}" if rank > 0 else "full_finetune"
        print(f"\n  [{label}] LoRA rank = {rank if rank > 0 else 'N/A (full)'} ...")

        model = EvoCRMModel(n_features, tasks=config["tasks"],
                             n_event_types=config["n_event_types"])

        if rank > 0:
            # Freeze backbone, apply LoRA
            for p in model.customer_tower.parameters():
                p.requires_grad = False
            for p in model.interaction_tower.parameters():
                p.requires_grad = False
            for p in model.hub.parameters():
                p.requires_grad = False
            model = apply_lora(model, rank=rank, target_modules=["heads"])

        metrics = quick_train_eval(
            model, train_loader, val_loader, test_loader, device, epochs,
        )
        results[label] = metrics

    _print_ablation_results("LoRA Rank Sweep", results)
    return results


# ============================================================
# ABLATION D: Single vs Multi-task
# ============================================================

def ablation_single_task(train_loader, val_loader, test_loader,
                          n_features, config, device) -> Dict:
    """
    Train each task independently vs joint multi-task.
    Paper Table 5: Multi-task benefit.
    """
    print("\n" + "=" * 60)
    print("ABLATION D: SINGLE-TASK vs MULTI-TASK")
    print("  (Does joint training help?)")
    print("=" * 60)

    results = {}
    epochs = config.get("ablation_epochs", 15)
    all_tasks = config["tasks"]

    # Multi-task (control)
    print("\n  [Control] Multi-task (all heads jointly) ...")
    model_mt = EvoCRMModel(n_features, tasks=all_tasks,
                            n_event_types=config["n_event_types"])
    results["multi_task"] = quick_train_eval(
        model_mt, train_loader, val_loader, test_loader, device, epochs,
    )

    # Single-task for each head
    for task_name, task_type in all_tasks.items():
        print(f"  [Single] {task_name} only ...")
        single_tasks = {task_name: task_type}
        model_st = EvoCRMModel(n_features, tasks=single_tasks,
                                n_event_types=config["n_event_types"])
        metrics = quick_train_eval(
            model_st, train_loader, val_loader, test_loader, device, epochs,
        )
        results[f"single_{task_name}"] = metrics

    _print_ablation_results("Single vs Multi-task", results)
    return results


# ============================================================
# PRINTING
# ============================================================

def _print_ablation_results(title: str, results: Dict):
    """Print ablation results table."""
    print(f"\n  {'─' * 55}")
    print(f"  {title} — Results Summary")
    print(f"  {'─' * 55}")

    # Collect all metric keys
    all_keys = set()
    for v in results.values():
        all_keys.update(k for k in v if k != "trainable_params")

    metric_keys = sorted(all_keys)

    # Header
    header = f"  {'Variant':<25}"
    for mk in metric_keys[:4]:  # First 4 metrics
        header += f" {mk:>12}"
    header += f" {'Params':>10}"
    print(header)
    print(f"  {'─' * (25 + 13 * min(len(metric_keys), 4) + 11)}")

    for variant, metrics in results.items():
        line = f"  {variant:<25}"
        for mk in metric_keys[:4]:
            val = metrics.get(mk, 0)
            line += f" {val:>12.4f}"
        params = metrics.get("trainable_params", 0)
        line += f" {params:>10,}"
        print(line)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run EvoCRM ablation studies")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/ablations/")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["tower_dropout", "hub_type", "lora_sweep",
                                 "single_task", "all"])
    parser.add_argument("--epochs", type=int, default=15,
                        help="Epochs per ablation variant (lower = faster)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_ds = EvoCRMDataset(args.data_dir, split="train")
    val_ds = EvoCRMDataset(args.data_dir, split="val")
    test_ds = EvoCRMDataset(args.data_dir, split="test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"Features: {train_ds.n_features}")

    # Determine active tasks
    active_tasks = {}
    for t in ["churn", "upsell", "early_adopter"]:
        if t in train_ds.targets:
            active_tasks[t] = "binary"
    for t in ["clv", "days_next_purchase"]:
        if t in train_ds.targets:
            active_tasks[t] = "regression"
    if "satisfaction_risk" in train_ds.targets:
        active_tasks["satisfaction_risk"] = "binary"

    n_event_types = train_ds.metadata.get("interaction_tower", {}).get("num_event_types", 10)

    config = {
        "tasks": active_tasks,
        "n_event_types": n_event_types,
        "ablation_epochs": args.epochs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # Run ablations
    ablations = {
        "tower_dropout": ablation_tower_dropout,
        "hub_type": ablation_hub_type,
        "lora_sweep": ablation_lora_sweep,
        "single_task": ablation_single_task,
    }

    to_run = ablations if args.ablation == "all" else {args.ablation: ablations[args.ablation]}

    for name, fn in to_run.items():
        print(f"\n{'#' * 60}")
        print(f"# RUNNING ABLATION: {name.upper()}")
        print(f"{'#' * 60}")

        t0 = time.time()
        result = fn(
            train_loader, val_loader, test_loader,
            train_ds.n_features, config, device,
        )
        elapsed = time.time() - t0
        result["_runtime_seconds"] = round(elapsed, 1)

        all_results[name] = result

        # Save per-ablation
        with open(output_dir / f"{name}_results.json", "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Saved: {output_dir / f'{name}_results.json'} ({elapsed:.0f}s)")

    # Save all
    with open(output_dir / "all_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("ALL ABLATIONS COMPLETE")
    print(f"  Results: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
