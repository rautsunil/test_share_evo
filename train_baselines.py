"""
EvoCRM Baseline Trainer
========================

Trains traditional ML baselines on EvoCRM adapter output for all 6 task heads.
These numbers form the comparison rows in your paper's Table 1.

Baselines:
    - XGBoost (gradient boosted trees)
    - Random Forest
    - LightGBM
    - Logistic Regression / Linear Regression (simple baseline)
    - Matrix Factorization (for recommendation task)

Usage:
    python train_baselines.py --data_dir ./evocrm_data/ --output_dir ./results/baselines/
    python train_baselines.py --data_dir ./evocrm_olist/ --output_dir ./results/baselines_olist/

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
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, mean_squared_error, mean_absolute_error, r2_score,
    average_precision_score, log_loss,
)

warnings.filterwarnings("ignore")

# Optional imports — gracefully degrade if not installed
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: lightgbm not installed. Install with: pip install lightgbm")


# ============================================================
# DATA LOADER
# ============================================================

class BaselineDataLoader:
    """Loads adapter output for baseline training."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> Dict[str, Any]:
        """Load features, targets, and splits."""
        print("=" * 60)
        print("LOADING DATA FOR BASELINES")
        print("=" * 60)

        # Load customer features
        feat_path = self.data_dir / "features" / "customer_tower_features.csv"
        features = pd.read_csv(feat_path)
        print(f"  Features: {features.shape}")

        # Load targets
        targets = pd.read_csv(self.data_dir / "targets.csv")
        print(f"  Targets: {targets.shape}")

        # Load splits
        splits = {}
        for split in ["train", "val", "test"]:
            path = self.data_dir / "splits" / f"{split}_user_ids.npy"
            splits[split] = np.load(path, allow_pickle=True)
            print(f"  {split}: {len(splits[split]):,} users")

        # Load metadata
        metadata = json.load(open(self.data_dir / "metadata.json"))

        # Merge features and targets
        merged = features.merge(targets, on="user_id", how="inner")
        print(f"  Merged: {len(merged):,} rows")

        # Identify feature columns
        cat_cols = metadata.get("customer_tower", {}).get("categorical_features", [])
        target_cols = [
            "churn", "clv", "upsell", "next_item_id",
            "early_adopter", "days_next_purchase",
        ]
        # Add bonus targets if present
        if "satisfaction_risk" in merged.columns:
            target_cols.append("satisfaction_risk")
        if "avg_review_score" in merged.columns:
            target_cols.append("avg_review_score")

        feature_cols = [
            c for c in merged.columns
            if c not in ["user_id"] + target_cols
        ]

        # Encode categoricals
        label_encoders = {}
        for col in cat_cols:
            if col in merged.columns:
                le = LabelEncoder()
                merged[col] = le.fit_transform(merged[col].astype(str))
                label_encoders[col] = le

        print(f"  Feature cols: {len(feature_cols)} | Target cols: {len(target_cols)}")
        print(f"  Categorical (encoded): {cat_cols}\n")

        return {
            "data": merged,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "cat_cols": cat_cols,
            "splits": splits,
            "metadata": metadata,
        }


# ============================================================
# TASK DEFINITIONS
# ============================================================

CLASSIFICATION_TASKS = {
    "churn": {
        "target": "churn",
        "type": "binary",
        "metrics": ["auc_roc", "f1", "precision", "recall", "accuracy", "avg_precision"],
        "primary_metric": "auc_roc",
    },
    "upsell": {
        "target": "upsell",
        "type": "binary",
        "metrics": ["auc_roc", "f1", "precision", "recall", "accuracy"],
        "primary_metric": "auc_roc",
    },
    "early_adopter": {
        "target": "early_adopter",
        "type": "binary",
        "metrics": ["auc_roc", "f1", "precision", "recall", "accuracy"],
        "primary_metric": "auc_roc",
    },
}

REGRESSION_TASKS = {
    "clv": {
        "target": "clv",
        "type": "regression",
        "metrics": ["rmse", "mae", "r2"],
        "primary_metric": "r2",
    },
    "days_next_purchase": {
        "target": "days_next_purchase",
        "type": "regression",
        "metrics": ["rmse", "mae", "r2"],
        "primary_metric": "rmse",
    },
}

# Optional bonus tasks
OPTIONAL_TASKS = {
    "satisfaction_risk": {
        "target": "satisfaction_risk",
        "type": "binary",
        "metrics": ["auc_roc", "f1"],
        "primary_metric": "auc_roc",
    },
}


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_classification_metrics(y_true, y_pred_proba, y_pred) -> Dict[str, float]:
    """Compute all classification metrics."""
    metrics = {}
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["auc_roc"] = 0.5

    metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
    metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    try:
        metrics["avg_precision"] = average_precision_score(y_true, y_pred_proba)
    except ValueError:
        metrics["avg_precision"] = 0.0

    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute all regression metrics."""
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ============================================================
# MODEL FACTORY
# ============================================================

def get_classification_models(seed: int = 42) -> Dict[str, Any]:
    """Return dictionary of classification models to train."""
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=seed, C=1.0, solver="lbfgs"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, eval_metric="logloss",
            verbosity=0,
        )

    if HAS_LGBM:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )

    return models


def get_regression_models(seed: int = 42) -> Dict[str, Any]:
    """Return dictionary of regression models to train."""
    models = {
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            random_state=seed, n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgboost"] = XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0,
        )

    if HAS_LGBM:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbose=-1,
        )

    return models


# ============================================================
# TRAINING ENGINE
# ============================================================

class BaselineTrainer:
    """Trains and evaluates all baselines for all tasks."""

    def __init__(self, data: Dict, output_dir: str, seed: int = 42):
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.all_results = {}

    def run_all(self):
        """Train baselines for all tasks."""
        df = self.data["data"]
        feat_cols = self.data["feature_cols"]
        splits = self.data["splits"]

        # Build train/val/test masks
        train_mask = df["user_id"].isin(splits["train"])
        val_mask = df["user_id"].isin(splits["val"])
        test_mask = df["user_id"].isin(splits["test"])

        X_train = df.loc[train_mask, feat_cols].values
        X_val = df.loc[val_mask, feat_cols].values
        X_test = df.loc[test_mask, feat_cols].values

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        # Replace NaN/inf after scaling
        for X in [X_train, X_val, X_test]:
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0

        # ---- Classification tasks ----
        for task_name, task_cfg in CLASSIFICATION_TASKS.items():
            target = task_cfg["target"]
            if target not in df.columns:
                print(f"  Skipping {task_name}: target '{target}' not in data")
                continue

            y_train = df.loc[train_mask, target].values
            y_val = df.loc[val_mask, target].values
            y_test = df.loc[test_mask, target].values

            # Check for valid binary labels
            if len(np.unique(y_train)) < 2:
                print(f"  Skipping {task_name}: only one class in training data")
                continue

            self._train_classification_task(
                task_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

        # ---- Regression tasks ----
        for task_name, task_cfg in REGRESSION_TASKS.items():
            target = task_cfg["target"]
            if target not in df.columns:
                print(f"  Skipping {task_name}: target '{target}' not in data")
                continue

            y_train = df.loc[train_mask, target].values.astype(float)
            y_val = df.loc[val_mask, target].values.astype(float)
            y_test = df.loc[test_mask, target].values.astype(float)

            self._train_regression_task(
                task_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

        # ---- Optional tasks ----
        for task_name, task_cfg in OPTIONAL_TASKS.items():
            target = task_cfg["target"]
            if target not in df.columns:
                continue
            y_train = df.loc[train_mask, target].values
            y_test = df.loc[test_mask, target].values
            if len(np.unique(y_train)) < 2:
                continue
            y_val = df.loc[val_mask, target].values
            self._train_classification_task(
                task_name, X_train, y_train, X_val, y_val, X_test, y_test
            )

        # ---- Save all results ----
        self._save_results()
        self._print_summary()

    def _train_classification_task(self, task_name, X_train, y_train,
                                    X_val, y_val, X_test, y_test):
        """Train all classification models for a single task."""
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_name.upper()} (Classification)")
        print(f"  Train: {len(y_train):,} | Positive rate: {y_train.mean():.1%}")
        print(f"{'=' * 60}")

        models = get_classification_models(self.seed)
        task_results = {}

        for model_name, model in models.items():
            print(f"\n  Training {model_name} ...", end=" ")
            t0 = time.time()

            try:
                model.fit(X_train, y_train)
                elapsed = time.time() - t0

                # Predict
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                # Compute metrics
                metrics = compute_classification_metrics(y_test, y_pred_proba, y_pred)
                metrics["train_time_sec"] = round(elapsed, 2)

                # Val metrics for model selection
                y_val_proba = model.predict_proba(X_val)[:, 1]
                y_val_pred = (y_val_proba >= 0.5).astype(int)
                val_metrics = compute_classification_metrics(y_val, y_val_proba, y_val_pred)
                metrics["val_auc_roc"] = val_metrics["auc_roc"]

                task_results[model_name] = metrics
                print(f"AUC={metrics['auc_roc']:.4f} | F1={metrics['f1']:.4f} | "
                      f"Time={elapsed:.1f}s")

            except Exception as e:
                print(f"FAILED: {e}")
                task_results[model_name] = {"error": str(e)}

        self.all_results[task_name] = task_results

        # Save per-task results
        task_path = self.output_dir / f"{task_name}_results.json"
        with open(task_path, "w") as f:
            json.dump(task_results, f, indent=2)

    def _train_regression_task(self, task_name, X_train, y_train,
                                X_val, y_val, X_test, y_test):
        """Train all regression models for a single task."""
        print(f"\n{'=' * 60}")
        print(f"TASK: {task_name.upper()} (Regression)")
        print(f"  Train: {len(y_train):,} | Mean: {y_train.mean():.2f} | "
              f"Std: {y_train.std():.2f}")
        print(f"{'=' * 60}")

        # Log-transform CLV for better regression performance
        log_transform = task_name == "clv"
        if log_transform:
            y_train_t = np.log1p(y_train)
            y_val_t = np.log1p(y_val)
            y_test_t = np.log1p(y_test)
        else:
            y_train_t, y_val_t, y_test_t = y_train, y_val, y_test

        models = get_regression_models(self.seed)
        task_results = {}

        for model_name, model in models.items():
            print(f"\n  Training {model_name} ...", end=" ")
            t0 = time.time()

            try:
                model.fit(X_train, y_train_t)
                elapsed = time.time() - t0

                y_pred_t = model.predict(X_test)

                # Inverse transform if needed
                if log_transform:
                    y_pred = np.expm1(y_pred_t)
                    y_pred = np.clip(y_pred, 0, None)
                else:
                    y_pred = y_pred_t

                metrics = compute_regression_metrics(y_test, y_pred)
                metrics["train_time_sec"] = round(elapsed, 2)

                # Val metrics
                y_val_pred_t = model.predict(X_val)
                y_val_pred = np.expm1(y_val_pred_t) if log_transform else y_val_pred_t
                val_metrics = compute_regression_metrics(y_val, y_val_pred)
                metrics["val_r2"] = val_metrics["r2"]

                task_results[model_name] = metrics
                print(f"RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f} | "
                      f"Time={elapsed:.1f}s")

            except Exception as e:
                print(f"FAILED: {e}")
                task_results[model_name] = {"error": str(e)}

        self.all_results[task_name] = task_results

        task_path = self.output_dir / f"{task_name}_results.json"
        with open(task_path, "w") as f:
            json.dump(task_results, f, indent=2)

    def _save_results(self):
        """Save consolidated results."""
        summary_path = self.output_dir / "all_baseline_results.json"
        with open(summary_path, "w") as f:
            json.dump(self.all_results, f, indent=2)
        print(f"\n  All results saved to: {summary_path}")

    def _print_summary(self):
        """Print paper-ready summary table."""
        print("\n" + "=" * 70)
        print("  BASELINE RESULTS SUMMARY (Paper Table 1 — Baseline Rows)")
        print("=" * 70)

        for task_name, models in self.all_results.items():
            task_cfg = {
                **CLASSIFICATION_TASKS, **REGRESSION_TASKS, **OPTIONAL_TASKS
            }.get(task_name, {})
            primary = task_cfg.get("primary_metric", "auc_roc")

            print(f"\n  {task_name.upper()} (primary: {primary})")
            print(f"  {'Model':<22} {'Primary':>10} {'Val':>10} {'Time':>8}")
            print(f"  {'-' * 52}")

            best_val = -999
            best_model = ""
            for model_name, metrics in models.items():
                if "error" in metrics:
                    print(f"  {model_name:<22} {'ERROR':>10}")
                    continue
                val = metrics.get(primary, 0)
                val_m = metrics.get(f"val_{primary}", metrics.get("val_auc_roc", 0))
                t = metrics.get("train_time_sec", 0)
                marker = ""
                if val_m > best_val:
                    best_val = val_m
                    best_model = model_name
                print(f"  {model_name:<22} {val:>10.4f} {val_m:>10.4f} {t:>7.1f}s")

            if best_model:
                print(f"  → Best: {best_model}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train EvoCRM baselines")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results/baselines/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loader = BaselineDataLoader(args.data_dir)
    data = loader.load()

    trainer = BaselineTrainer(data, args.output_dir, args.seed)
    trainer.run_all()


if __name__ == "__main__":
    main()
