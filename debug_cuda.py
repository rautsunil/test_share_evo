"""
EvoCRM CUDA Debug Diagnostic
==============================

Run this BEFORE training to inspect your actual data for every
known cause of 'CUDA device-side assert triggered'.

Usage:
    python debug_cuda.py --data_dir ./data/olist/
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def run_diagnostics(data_dir: str):
    data_dir = Path(data_dir)
    print("=" * 60)
    print("  CUDA ASSERT DIAGNOSTIC")
    print("=" * 60)
    
    issues_found = 0

    # ============================================================
    # CHECK 1: Target values
    # ============================================================
    print("\n[CHECK 1] TARGET VALUES")
    targets = pd.read_csv(data_dir / "targets.csv")
    
    binary_cols = ["churn", "upsell", "early_adopter", "satisfaction_risk"]
    for col in binary_cols:
        if col not in targets.columns:
            continue
        vals = targets[col]
        nan_count = vals.isna().sum()
        unique = vals.dropna().unique()
        non_binary = [v for v in unique if v not in [0, 1, 0.0, 1.0]]
        
        print(f"  {col}:")
        print(f"    dtype={vals.dtype}, NaN={nan_count}, unique={sorted(unique)[:10]}")
        if nan_count > 0:
            print(f"    ✗ HAS {nan_count} NaN VALUES — THIS CAUSES CUDA ASSERT IN BCE")
            issues_found += 1
        if non_binary:
            print(f"    ✗ NON-BINARY VALUES: {non_binary[:5]} — THIS CAUSES CUDA ASSERT IN BCE")
            issues_found += 1
        if nan_count == 0 and not non_binary:
            print(f"    ✓ OK")

    reg_cols = ["clv", "days_next_purchase"]
    for col in reg_cols:
        if col not in targets.columns:
            continue
        vals = targets[col]
        nan_count = vals.isna().sum()
        inf_count = np.isinf(vals.values.astype(float)).sum()
        print(f"  {col}:")
        print(f"    dtype={vals.dtype}, NaN={nan_count}, Inf={inf_count}, "
              f"range=[{vals.min()}, {vals.max()}]")
        if nan_count > 0 or inf_count > 0:
            print(f"    ✗ HAS NaN/Inf — WILL CAUSE LOSS EXPLOSION")
            issues_found += 1
        else:
            print(f"    ✓ OK")

    # ============================================================
    # CHECK 2: Feature values
    # ============================================================
    print("\n[CHECK 2] FEATURE VALUES")
    features = pd.read_csv(data_dir / "features" / "customer_tower_features.csv")
    
    num_cols = features.select_dtypes(include=[np.number]).columns
    total_nan = features[num_cols].isna().sum().sum()
    total_inf = np.isinf(features[num_cols].values.astype(float)).sum()
    
    print(f"  Shape: {features.shape}")
    print(f"  Total NaN in numerics: {total_nan}")
    print(f"  Total Inf in numerics: {total_inf}")
    
    # Check for extremely large values
    for col in num_cols:
        mx = features[col].abs().max()
        if mx > 1e10:
            print(f"  ✗ {col}: max abs value = {mx:.2e} — MAY CAUSE OVERFLOW")
            issues_found += 1
    
    # Check dtypes — object columns that should be numeric
    obj_cols = features.select_dtypes(include=["object"]).columns.tolist()
    obj_cols = [c for c in obj_cols if c != "user_id"]
    if obj_cols:
        print(f"  ⚠ String columns (will be excluded from tensor): {obj_cols}")
    
    if total_nan == 0 and total_inf == 0:
        print(f"  ✓ OK")

    # ============================================================
    # CHECK 3: Interaction sequences — embedding index bounds
    # ============================================================
    print("\n[CHECK 3] INTERACTION SEQUENCE INDICES")
    seq_path = data_dir / "features" / "interaction_sequences.json"
    
    if not seq_path.exists():
        print(f"  ✗ interaction_sequences.json NOT FOUND")
        issues_found += 1
    else:
        with open(seq_path) as f:
            seqs = json.load(f)
        
        max_event_id = 0
        max_product_id = 0
        negative_ids = 0
        non_numeric = 0
        nan_in_seqs = 0
        
        for uid, events in seqs.items():
            for event in events:
                if len(event) != 3:
                    continue
                try:
                    eid = int(float(event[0]))
                    pid = int(float(event[1]))
                    td = float(event[2])
                    
                    max_event_id = max(max_event_id, eid)
                    max_product_id = max(max_product_id, pid)
                    
                    if eid < 0 or pid < 0:
                        negative_ids += 1
                    if np.isnan(td):
                        nan_in_seqs += 1
                except (ValueError, TypeError):
                    non_numeric += 1
        
        # Load metadata to compare
        metadata = json.load(open(data_dir / "metadata.json"))
        declared_event_types = metadata.get("interaction_tower", {}).get("num_event_types", 10)
        
        print(f"  Total sequences: {len(seqs):,}")
        print(f"  Max event_type_id: {max_event_id}")
        print(f"  Declared num_event_types: {declared_event_types}")
        print(f"  Max product_id: {max_product_id}")
        print(f"  Negative IDs: {negative_ids}")
        print(f"  Non-numeric values: {non_numeric}")
        print(f"  NaN in time_deltas: {nan_in_seqs}")
        
        if max_event_id > declared_event_types:
            print(f"  ✗ max_event_id ({max_event_id}) > declared ({declared_event_types})")
            print(f"    THIS CAUSES EMBEDDING INDEX OUT OF BOUNDS → CUDA ASSERT")
            issues_found += 1
        
        if max_product_id > 100000:
            print(f"  ⚠ max_product_id is {max_product_id:,} — embedding table must be this big")
        
        if negative_ids > 0:
            print(f"  ✗ {negative_ids} NEGATIVE IDs — EMBEDDING WILL CRASH")
            issues_found += 1
            
        if non_numeric > 0:
            print(f"  ✗ {non_numeric} NON-NUMERIC values in sequences")
            issues_found += 1
            
        if nan_in_seqs > 0:
            print(f"  ✗ {nan_in_seqs} NaN time_deltas in sequences")
            issues_found += 1

        if max_event_id <= declared_event_types and negative_ids == 0:
            print(f"  ✓ Event IDs OK")
        if max_product_id < 100000 and negative_ids == 0:
            print(f"  ✓ Product IDs OK (max={max_product_id})")

    # ============================================================
    # CHECK 4: Feature-target user_id alignment
    # ============================================================
    print("\n[CHECK 4] USER ID ALIGNMENT")
    feat_uids = set(features["user_id"].values)
    tgt_uids = set(targets["user_id"].values)
    
    overlap = feat_uids & tgt_uids
    feat_only = feat_uids - tgt_uids
    tgt_only = tgt_uids - feat_uids
    
    print(f"  Features: {len(feat_uids):,} users")
    print(f"  Targets:  {len(tgt_uids):,} users")
    print(f"  Overlap:  {len(overlap):,}")
    
    if feat_only:
        print(f"  ⚠ {len(feat_only)} users in features but not targets")
    if tgt_only:
        print(f"  ⚠ {len(tgt_only)} users in targets but not features")

    # Check splits
    for split in ["train", "val", "test"]:
        sp = np.load(data_dir / "splits" / f"{split}_user_ids.npy", allow_pickle=True)
        in_both = set(sp) & overlap
        outside = set(sp) - overlap
        print(f"  {split}: {len(sp):,} users, {len(in_both):,} have both features+targets, "
              f"{len(outside)} orphaned")
        if outside:
            print(f"    ✗ {len(outside)} users in {split} split have NO features or targets")
            issues_found += 1

    # ============================================================
    # CHECK 5: Simulate tensor creation
    # ============================================================
    print("\n[CHECK 5] SIMULATED TENSOR CREATION")
    try:
        import torch
        
        # Simulate what EvoCRMDataset does
        merged = features.merge(targets, on="user_id", how="inner")
        print(f"  Merged: {len(merged):,} rows")
        
        # Check binary targets after merge
        for col in binary_cols:
            if col not in merged.columns:
                continue
            vals = pd.to_numeric(merged[col], errors="coerce").fillna(0).values
            vals = np.clip(vals, 0, 1).astype(np.float32)
            t = torch.FloatTensor(vals)
            has_nan = torch.isnan(t).any().item()
            print(f"  {col} tensor: shape={t.shape}, NaN={has_nan}, "
                  f"range=[{t.min():.4f}, {t.max():.4f}]")
            if has_nan:
                print(f"    ✗ STILL HAS NaN AFTER CONVERSION!")
                issues_found += 1
        
        # Check features tensor
        tgt_cols = ["churn", "clv", "upsell", "next_item_id", "early_adopter",
                    "days_next_purchase", "satisfaction_risk", "avg_review_score"]
        feat_cols = [c for c in merged.columns if c not in ["user_id"] + tgt_cols]
        
        feat_vals = merged[feat_cols].values
        # Check for non-numeric columns
        non_num = []
        for i, col in enumerate(feat_cols):
            try:
                float(merged[col].iloc[0])
            except (ValueError, TypeError):
                non_num.append(col)
        
        if non_num:
            print(f"  ✗ NON-NUMERIC feature columns: {non_num}")
            print(f"    THESE CANNOT BE CONVERTED TO FloatTensor — WILL CRASH")
            issues_found += 1
        else:
            feat_tensor = torch.FloatTensor(feat_vals.astype(np.float32))
            feat_tensor = torch.nan_to_num(feat_tensor)
            print(f"  Feature tensor: shape={feat_tensor.shape}, "
                  f"NaN={torch.isnan(feat_tensor).any().item()}, "
                  f"Inf={torch.isinf(feat_tensor).any().item()}")
        
        # Simulate sequence tensor with actual embedding bounds
        if seq_path.exists():
            print(f"\n  Embedding bounds test:")
            print(f"    nn.Embedding(num_event_types+1={declared_event_types + 1}, 64)")
            print(f"    nn.Embedding(n_products+1={max_product_id + 1}, 64)")
            
            # Test a few sequences
            test_event = torch.LongTensor([max_event_id])
            test_prod = torch.LongTensor([max_product_id])
            
            evt_emb = torch.nn.Embedding(declared_event_types + 1, 64, padding_idx=0)
            prod_emb = torch.nn.Embedding(max_product_id + 1, 64, padding_idx=0)
            
            try:
                evt_emb(test_event)
                print(f"    ✓ Event embedding({max_event_id}) works")
            except IndexError as e:
                print(f"    ✗ Event embedding FAILS: {e}")
                issues_found += 1
            
            try:
                prod_emb(test_prod)
                print(f"    ✓ Product embedding({max_product_id}) works")
            except IndexError as e:
                print(f"    ✗ Product embedding FAILS: {e}")
                issues_found += 1
                
    except ImportError:
        print("  ⚠ PyTorch not available — skipping tensor simulation")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    if issues_found == 0:
        print("  ✓ NO ISSUES FOUND IN DATA")
        print("  If CUDA assert still happens, the bug is in model code.")
        print("  Run with: CUDA_LAUNCH_BLOCKING=1 python train_evocrm.py ...")
        print("  and paste the FULL traceback here.")
    else:
        print(f"  ✗ {issues_found} ISSUES FOUND — FIX THESE BEFORE TRAINING")
    print("=" * 60)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./data/olist/")
    args = p.parse_args()
    run_diagnostics(args.data_dir)
