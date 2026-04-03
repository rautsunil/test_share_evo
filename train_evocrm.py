"""
EvoCRM Training Script — 3-Phase Training Pipeline
=====================================================

Trains the EvoCRM hub-and-spoke model in three phases:
    Phase 1 (towers):   Pre-train each tower independently
    Phase 2 (hub):      Pre-train the Hub with frozen towers
    Phase 3 (finetune): Fine-tune task heads with LoRA adapters

Usage:
    # Phase 1: Pre-train towers
    python train_evocrm.py --data_dir ./evocrm_data/ --phase towers --epochs 50

    # Phase 2: Pre-train hub
    python train_evocrm.py --data_dir ./evocrm_data/ --phase hub --epochs 30 \
        --tower_ckpt checkpoints/towers_best.pt

    # Phase 3: Fine-tune with LoRA
    python train_evocrm.py --data_dir ./evocrm_data/ --phase finetune --epochs 20 \
        --hub_ckpt checkpoints/hub_best.pt --use_lora --lora_rank 8

    # Full pipeline (all 3 phases sequentially)
    python train_evocrm.py --data_dir ./evocrm_data/ --phase all --epochs 50,30,20

Author: EvoCRM Team
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")


# ============================================================
# DATASET
# ============================================================

class EvoCRMDataset(Dataset):
    """PyTorch Dataset for EvoCRM adapter output."""

    def __init__(self, data_dir: str, split: str = "train",
                 max_seq_length: int = 128):
        self.data_dir = Path(data_dir)
        self.max_seq_length = max_seq_length

        # Load features
        features_df = pd.read_csv(
            self.data_dir / "features" / "customer_tower_features.csv"
        )
        targets_df = pd.read_csv(self.data_dir / "targets.csv")

        # Load split user IDs
        split_ids = np.load(
            self.data_dir / "splits" / f"{split}_user_ids.npy",
            allow_pickle=True,
        )

        # Load metadata
        self.metadata = json.load(open(self.data_dir / "metadata.json"))
        cat_cols = self.metadata["customer_tower"]["categorical_features"]

        # Merge and filter to split
        merged = features_df.merge(targets_df, on="user_id", how="inner")
        merged = merged[merged["user_id"].isin(split_ids)].reset_index(drop=True)

        # Encode categoricals
        self.cat_encoders = {}
        for col in cat_cols:
            if col in merged.columns:
                merged[col] = merged[col].astype(str)
                codes, uniques = pd.factorize(merged[col])
                merged[col] = codes
                self.cat_encoders[col] = uniques

        # Separate features
        target_cols = [
            "churn", "clv", "upsell", "next_item_id",
            "early_adopter", "days_next_purchase",
        ]
        optional_targets = ["satisfaction_risk", "avg_review_score"]
        for t in optional_targets:
            if t in merged.columns:
                target_cols.append(t)

        feature_cols = [
            c for c in merged.columns
            if c not in ["user_id"] + target_cols
        ]

        # Store as tensors
        self.user_ids = merged["user_id"].values
        self.features = torch.FloatTensor(
            merged[feature_cols].values.astype(np.float32)
        )
        self.features = torch.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)

        # Targets
        self.targets = {}
        for col in ["churn", "upsell", "early_adopter"]:
            if col in merged.columns:
                self.targets[col] = torch.FloatTensor(merged[col].values)
        for col in ["clv", "days_next_purchase"]:
            if col in merged.columns:
                self.targets[col] = torch.FloatTensor(merged[col].values)
        if "satisfaction_risk" in merged.columns:
            self.targets["satisfaction_risk"] = torch.FloatTensor(
                merged["satisfaction_risk"].values
            )

        # Load interaction sequences
        seq_path = self.data_dir / "features" / "interaction_sequences.json"
        if seq_path.exists():
            with open(seq_path) as f:
                raw_seqs = json.load(f)
            self.sequences = self._build_sequence_tensors(raw_seqs, merged["user_id"].values)
        else:
            # Dummy sequences if not available
            self.sequences = torch.zeros(len(merged), max_seq_length, 3)

        self.n_features = self.features.shape[1]

    def _build_sequence_tensors(self, raw_seqs, user_ids):
        """Convert JSON sequences to padded tensor."""
        tensors = []
        for uid in user_ids:
            key = str(uid)
            if key in raw_seqs:
                seq = raw_seqs[key]
                # Truncate
                seq = seq[-self.max_seq_length:]
                # Pad
                while len(seq) < self.max_seq_length:
                    seq.insert(0, [0, 0, 0.0])
                tensors.append(seq)
            else:
                tensors.append([[0, 0, 0.0]] * self.max_seq_length)

        return torch.FloatTensor(tensors)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "features": self.features[idx],
            "sequences": self.sequences[idx],
            "targets": {k: v[idx] for k, v in self.targets.items()},
        }


# ============================================================
# MODEL COMPONENTS
# ============================================================

class CustomerTower(nn.Module):
    """Customer Tower: MLP with BatchNorm → embedding."""

    def __init__(self, input_dim: int, embed_dim: int = 256,
                 hidden_dims: List[int] = [512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.2),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, embed_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class InteractionTower(nn.Module):
    """Interaction Tower: Transformer over event sequences."""

    def __init__(self, embed_dim: int = 256, n_event_types: int = 20,
                 n_heads: int = 4, n_layers: int = 2, max_seq_len: int = 128):
        super().__init__()
        self.event_embed = nn.Embedding(n_event_types + 1, 64, padding_idx=0)
        self.product_embed = nn.Embedding(50000, 64, padding_idx=0)
        self.time_proj = nn.Linear(1, 64)
        self.input_proj = nn.Linear(192, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, sequences):
        # sequences: (batch, seq_len, 3) → [event_type_id, product_id, time_delta]
        event_ids = sequences[:, :, 0].long().clamp(0, self.event_embed.num_embeddings - 1)
        product_ids = sequences[:, :, 1].long().clamp(0, self.product_embed.num_embeddings - 1)
        time_deltas = sequences[:, :, 2:3]

        event_emb = self.event_embed(event_ids)
        prod_emb = self.product_embed(product_ids)
        time_emb = self.time_proj(time_deltas)

        combined = torch.cat([event_emb, prod_emb, time_emb], dim=-1)
        projected = self.input_proj(combined)

        # Mask padding (all zeros)
        mask = (sequences.sum(dim=-1) == 0)
        encoded = self.transformer(projected, src_key_padding_mask=mask)

        # Pool over sequence dimension
        pooled = self.pool(encoded.transpose(1, 2)).squeeze(-1)
        return pooled


class Hub(nn.Module):
    """Hub: Cross-modal fusion with gating."""

    def __init__(self, tower_dim: int = 256, hub_dim: int = 512,
                 n_heads: int = 4):
        super().__init__()
        self.customer_proj = nn.Linear(tower_dim, hub_dim)
        self.interaction_proj = nn.Linear(tower_dim, hub_dim)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hub_dim, num_heads=n_heads, batch_first=True, dropout=0.1
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(hub_dim * 2, hub_dim),
            nn.Sigmoid(),
        )

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(hub_dim, hub_dim),
            nn.LayerNorm(hub_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hub_dim, hub_dim),
        )

    def forward(self, customer_emb, interaction_emb):
        c = self.customer_proj(customer_emb).unsqueeze(1)
        i = self.interaction_proj(interaction_emb).unsqueeze(1)

        # Stack as sequence for cross-attention
        combined = torch.cat([c, i], dim=1)  # (batch, 2, hub_dim)

        # Self-attention across modalities
        attn_out, _ = self.cross_attn(combined, combined, combined)

        # Pool
        pooled = attn_out.mean(dim=1)  # (batch, hub_dim)

        # Gating
        gate_input = torch.cat([c.squeeze(1), i.squeeze(1)], dim=-1)
        gate_weight = self.gate(gate_input)
        gated = pooled * gate_weight

        return self.fusion(gated)


class MultiTaskHead(nn.Module):
    """Multi-task prediction heads."""

    def __init__(self, input_dim: int = 512, tasks: Dict[str, str] = None):
        super().__init__()
        self.heads = nn.ModuleDict()

        default_tasks = {
            "churn": "binary", "clv": "regression", "upsell": "binary",
            "early_adopter": "binary", "days_next_purchase": "regression",
            "satisfaction_risk": "binary",
        }
        tasks = tasks or default_tasks

        for task_name, task_type in tasks.items():
            out_dim = 1
            self.heads[task_name] = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, out_dim),
            )

        self.task_types = tasks

    def forward(self, hub_output, task_name: str = None):
        results = {}
        task_list = [task_name] if task_name else list(self.heads.keys())

        for name in task_list:
            if name not in self.heads:
                continue
            logits = self.heads[name](hub_output).squeeze(-1)
            if self.task_types.get(name) == "binary":
                results[name] = torch.sigmoid(logits)
            else:
                results[name] = logits

        return results


class EvoCRMModel(nn.Module):
    """Complete EvoCRM model."""

    def __init__(self, n_features: int, embed_dim: int = 256,
                 hub_dim: int = 512, n_event_types: int = 20,
                 tasks: Dict[str, str] = None):
        super().__init__()
        self.customer_tower = CustomerTower(n_features, embed_dim)
        self.interaction_tower = InteractionTower(embed_dim, n_event_types)
        self.hub = Hub(embed_dim, hub_dim)
        self.task_heads = MultiTaskHead(hub_dim, tasks)

    def forward(self, features, sequences, task_name=None):
        c_emb = self.customer_tower(features)
        i_emb = self.interaction_tower(sequences)
        hub_out = self.hub(c_emb, i_emb)
        predictions = self.task_heads(hub_out, task_name)
        return predictions, hub_out


# ============================================================
# LORA ADAPTER
# ============================================================

class LoRALinear(nn.Module):
    """LoRA adapter for a linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Freeze original
        for p in self.original.parameters():
            p.requires_grad = False

        # Low-rank decomposition
        self.lora_A = nn.Parameter(torch.randn(original.in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base + lora


def apply_lora(model: nn.Module, rank: int = 8, target_modules: List[str] = None):
    """Apply LoRA to specified linear layers."""
    target_modules = target_modules or ["fusion", "heads"]
    n_lora = 0
    n_total = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            n_total += 1
            if any(t in name for t in target_modules):
                parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = dict(model.named_modules())[parent_name] if parent_name else model
                lora_layer = LoRALinear(module, rank=rank)
                setattr(parent, attr_name, lora_layer)
                n_lora += 1

    print(f"  Applied LoRA (rank={rank}) to {n_lora}/{n_total} linear layers")
    return model


# ============================================================
# LOSS FUNCTIONS
# ============================================================

class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with uncertainty weighting."""

    def __init__(self, task_types: Dict[str, str]):
        super().__init__()
        self.task_types = task_types
        # Learnable uncertainty weights (Kendall et al., 2018)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in task_types
        })

    def forward(self, predictions: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        losses = {}

        for task_name, pred in predictions.items():
            if task_name not in targets:
                continue

            target = targets[task_name]
            task_type = self.task_types.get(task_name, "binary")

            if task_type == "binary":
                loss = nn.functional.binary_cross_entropy(
                    pred.clamp(1e-7, 1 - 1e-7), target, reduction="mean"
                )
            else:
                # Huber loss for regression (robust to outliers)
                loss = nn.functional.huber_loss(pred, target, reduction="mean", delta=1.0)

            # Uncertainty weighting
            precision = torch.exp(-self.log_vars[task_name])
            weighted_loss = precision * loss + self.log_vars[task_name]

            total_loss = total_loss + weighted_loss
            losses[task_name] = loss.item()

        return total_loss, losses


# ============================================================
# TRAINING ENGINE
# ============================================================

class EvoCRMTrainer:
    """3-phase training engine."""

    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.history = {"train": [], "val": []}

    def train_phase(self, phase: str, epochs: int, lr: float,
                    freeze_towers: bool = False, freeze_hub: bool = False):
        """Run a single training phase."""
        print(f"\n{'=' * 60}")
        print(f"PHASE: {phase.upper()} | Epochs: {epochs} | LR: {lr}")
        print(f"  Freeze towers: {freeze_towers} | Freeze hub: {freeze_hub}")
        print(f"{'=' * 60}")

        # Freeze/unfreeze
        if freeze_towers:
            for p in self.model.customer_tower.parameters():
                p.requires_grad = False
            for p in self.model.interaction_tower.parameters():
                p.requires_grad = False
        if freeze_hub:
            for p in self.model.hub.parameters():
                p.requires_grad = False

        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} "
              f"({trainable/total:.1%})")

        # Loss and optimizer
        task_types = self.model.task_heads.task_types
        criterion = MultiTaskLoss(task_types).to(self.device)
        optimizer = optim.AdamW(
            list(self.model.parameters()) + list(criterion.parameters()),
            lr=lr, weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        ckpt_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs + 1):
            # ---- Train ----
            self.model.train()
            epoch_losses = {}
            n_batches = 0

            for batch in self.train_loader:
                features = batch["features"].to(self.device)
                sequences = batch["sequences"].to(self.device)
                targets = {
                    k: v.to(self.device)
                    for k, v in batch["targets"].items()
                }

                optimizer.zero_grad()
                predictions, _ = self.model(features, sequences)
                loss, task_losses = criterion(predictions, targets)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                for k, v in task_losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                n_batches += 1

            scheduler.step()

            # Average losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            # ---- Validate ----
            val_loss, val_task_losses = self._validate(criterion)

            self.history["train"].append(epoch_losses)
            self.history["val"].append(val_task_losses)

            # ---- Logging ----
            train_str = " | ".join(f"{k}={v:.4f}" for k, v in epoch_losses.items())
            val_str = " | ".join(f"{k}={v:.4f}" for k, v in val_task_losses.items())

            if epoch % max(1, epochs // 10) == 0 or epoch == 1:
                print(f"  Epoch {epoch:>3}/{epochs} | Train: {train_str}")
                print(f"  {'':>14} | Val:   {val_str}")

            # ---- Checkpoint ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = ckpt_dir / f"{phase}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "val_loss": val_loss,
                    "phase": phase,
                }, ckpt_path)

        print(f"\n  Best val loss: {best_val_loss:.4f}")
        print(f"  Checkpoint: {ckpt_dir / f'{phase}_best.pt'}")

        # Unfreeze everything for next phase
        for p in self.model.parameters():
            p.requires_grad = True

        return best_val_loss

    @torch.no_grad()
    def _validate(self, criterion):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        task_losses = {}
        n_batches = 0

        for batch in self.val_loader:
            features = batch["features"].to(self.device)
            sequences = batch["sequences"].to(self.device)
            targets = {k: v.to(self.device) for k, v in batch["targets"].items()}

            predictions, _ = self.model(features, sequences)
            loss, batch_task_losses = criterion(predictions, targets)

            total_loss += loss.item()
            for k, v in batch_task_losses.items():
                task_losses[k] = task_losses.get(k, 0) + v
            n_batches += 1

        for k in task_losses:
            task_losses[k] /= max(n_batches, 1)

        return total_loss / max(n_batches, 1), task_losses

    def save_history(self, path: str):
        """Save training history."""
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_pipeline(args):
    """Run the training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load data
    print("\nLoading datasets ...")
    train_ds = EvoCRMDataset(args.data_dir, split="train")
    val_ds = EvoCRMDataset(args.data_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    print(f"  Features: {train_ds.n_features} | Seq length: {args.max_seq_len}")

    # Determine active tasks
    active_tasks = {}
    for task_name in ["churn", "upsell", "early_adopter"]:
        if task_name in train_ds.targets:
            active_tasks[task_name] = "binary"
    for task_name in ["clv", "days_next_purchase"]:
        if task_name in train_ds.targets:
            active_tasks[task_name] = "regression"
    if "satisfaction_risk" in train_ds.targets:
        active_tasks["satisfaction_risk"] = "binary"

    print(f"  Active tasks: {list(active_tasks.keys())}")

    # Build model
    n_event_types = train_ds.metadata.get("interaction_tower", {}).get("num_event_types", 10)
    model = EvoCRMModel(
        n_features=train_ds.n_features,
        embed_dim=args.embed_dim,
        hub_dim=args.hub_dim,
        n_event_types=n_event_types,
        tasks=active_tasks,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Load checkpoint if provided
    if args.tower_ckpt and os.path.exists(args.tower_ckpt):
        ckpt = torch.load(args.tower_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Loaded tower checkpoint: {args.tower_ckpt}")

    if args.hub_ckpt and os.path.exists(args.hub_ckpt):
        ckpt = torch.load(args.hub_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"  Loaded hub checkpoint: {args.hub_ckpt}")

    config = {
        "checkpoint_dir": args.checkpoint_dir,
        "data_dir": args.data_dir,
    }

    trainer = EvoCRMTrainer(model, train_loader, val_loader, config, device)

    # Parse epochs per phase
    if args.phase == "all":
        epoch_list = [int(e) for e in args.epochs.split(",")]
        if len(epoch_list) < 3:
            epoch_list = epoch_list + [epoch_list[-1]] * (3 - len(epoch_list))
        phases = [
            ("towers", epoch_list[0], args.lr, False, False),
            ("hub", epoch_list[1], args.lr * 0.5, True, False),
            ("finetune", epoch_list[2], args.lr * 0.1, False, False),
        ]
    elif args.phase == "towers":
        phases = [("towers", int(args.epochs), args.lr, False, False)]
    elif args.phase == "hub":
        phases = [("hub", int(args.epochs), args.lr * 0.5, True, False)]
    elif args.phase == "finetune":
        phases = [("finetune", int(args.epochs), args.lr * 0.1, False, False)]
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    # Apply LoRA for finetune phase
    if args.use_lora and (args.phase in ["finetune", "all"]):
        model = apply_lora(model, rank=args.lora_rank)
        model = model.to(device)
        trainer.model = model

    # Run phases
    for phase_name, epochs, lr, freeze_t, freeze_h in phases:
        if phase_name == "finetune" and args.use_lora and args.phase == "all":
            model = apply_lora(model, rank=args.lora_rank)
            model = model.to(device)
            trainer.model = model

        trainer.train_phase(phase_name, epochs, lr, freeze_t, freeze_h)

    # Save final model and history
    final_path = Path(args.checkpoint_dir) / "evocrm_final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "n_features": train_ds.n_features,
            "embed_dim": args.embed_dim,
            "hub_dim": args.hub_dim,
            "n_event_types": n_event_types,
            "tasks": active_tasks,
        },
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")

    history_path = Path(args.checkpoint_dir) / "training_history.json"
    trainer.save_history(str(history_path))
    print(f"Training history saved to: {history_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train EvoCRM model")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--phase", type=str, default="all",
                        choices=["towers", "hub", "finetune", "all"])
    parser.add_argument("--epochs", type=str, default="50,30,20",
                        help="Epochs (single int or comma-separated for 'all')")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hub_dim", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--tower_ckpt", type=str, default=None)
    parser.add_argument("--hub_ckpt", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_pipeline(args)


if __name__ == "__main__":
    main()
