"""
Olist Brazilian E-Commerce → EvoCRM Data Adapter Pipeline
============================================================

Converts the Olist public dataset (8 relational tables) into EvoCRM's
4-table format with all 6 task head targets.

ADVANTAGES OVER RETAILROCKET:
    ✓ Real prices and payment amounts (no estimation needed)
    ✓ Real customer locations (city, state, zip)
    ✓ Real product categories with English translations
    ✓ Review scores (1-5) — usable as quality signal
    ✓ Delivery performance data (estimated vs actual)
    ✓ Multiple payment methods per order

LIMITATIONS:
    ✗ No web browsing data (only order-level events)
    ✗ No campaign/marketing data
    ✗ ~100K orders, ~96K unique customers (moderate size)
    ✗ Marketplace model (multi-seller) — different from single-brand DTC

Olist Source Files (8 CSV + 1 translation):
    - olist_customers_dataset.csv
    - olist_orders_dataset.csv
    - olist_order_items_dataset.csv
    - olist_order_payments_dataset.csv
    - olist_order_reviews_dataset.csv
    - olist_products_dataset.csv
    - olist_sellers_dataset.csv
    - olist_geolocation_dataset.csv
    - product_category_name_translation.csv

Usage:
    python olist_adapter.py --data_dir ./olist_raw/ --output_dir ./evocrm_olist/

Author: EvoCRM Team
Dataset: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
"""

import os
import sys
import json
import hashlib
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class OlistAdapterConfig:
    """Configuration for Olist → EvoCRM conversion."""

    # --- Input ---
    data_dir: str = "./olist_raw/"

    # --- Output ---
    output_dir: str = "./evocrm_olist/"

    # --- Order Status Filter ---
    # Only keep delivered/shipped orders (exclude cancelled, etc.)
    valid_order_statuses: tuple = ("delivered", "shipped", "invoiced", "processing")

    # --- User Filtering ---
    min_orders_per_user: int = 1  # Olist has many single-order customers

    # --- Session / Interaction Synthesis ---
    # Olist has no web behavior; we synthesize an "order journey" sequence
    # from: order_placed → payment → shipped → delivered → reviewed
    enable_order_journey_sequences: bool = True

    # --- Churn Definition ---
    churn_window_days: int = 90
    churn_gap_days: int = 30

    # --- CLV ---
    clv_observation_months: int = 6

    # --- Upsell ---
    upsell_aov_increase_pct: float = 0.20

    # --- Early Adopter ---
    early_adopter_days: int = 30

    # --- Next Purchase ---
    next_purchase_max_days: int = 365

    # --- Data Splits ---
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # --- Review Score Thresholds ---
    satisfied_threshold: int = 4  # review_score >= 4 is "satisfied"
    dissatisfied_threshold: int = 2  # review_score <= 2 is "dissatisfied"

    # --- Reproducibility ---
    seed: int = 42


# ============================================================
# STEP 1: RAW DATA LOADER
# ============================================================

class OlistLoader:
    """Loads and validates all 8+1 Olist CSV files."""

    REQUIRED_FILES = {
        "customers": "olist_customers_dataset.csv",
        "orders": "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
    }

    OPTIONAL_FILES = {
        "geolocation": "olist_geolocation_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }

    def __init__(self, config: OlistAdapterConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all Olist files with validation."""
        print("=" * 60)
        print("STEP 1: LOADING OLIST DATASET (8 tables)")
        print("=" * 60)

        data = {}

        # Required files
        for key, filename in self.REQUIRED_FILES.items():
            path = self.data_dir / filename
            if not path.exists():
                raise FileNotFoundError(
                    f"{filename} not found at {path}.\n"
                    f"Download from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
                )
            df = pd.read_csv(path)
            data[key] = df
            print(f"  ✓ {filename}: {len(df):,} rows × {len(df.columns)} cols")

        # Optional files
        for key, filename in self.OPTIONAL_FILES.items():
            path = self.data_dir / filename
            if path.exists():
                data[key] = pd.read_csv(path)
                print(f"  ✓ {filename}: {len(data[key]):,} rows (optional)")
            else:
                data[key] = pd.DataFrame()
                print(f"  ⚠ {filename}: not found (optional, skipping)")

        # Parse datetime columns
        dt_cols = {
            "orders": [
                "order_purchase_timestamp", "order_approved_at",
                "order_delivered_carrier_date", "order_delivered_customer_date",
                "order_estimated_delivery_date",
            ],
            "reviews": ["review_creation_date", "review_answer_timestamp"],
        }
        for table, cols in dt_cols.items():
            for col in cols:
                if col in data[table].columns:
                    data[table][col] = pd.to_datetime(
                        data[table][col], errors="coerce"
                    )

        # Translate product categories to English
        if not data["category_translation"].empty and not data["products"].empty:
            data["products"] = data["products"].merge(
                data["category_translation"],
                on="product_category_name",
                how="left",
            )
            # Use English name where available, else original
            data["products"]["category_english"] = (
                data["products"]["product_category_name_english"]
                .fillna(data["products"]["product_category_name"])
            )
            n_translated = data["products"]["product_category_name_english"].notna().sum()
            print(f"  → Translated {n_translated:,} / {len(data['products']):,} product categories")

        self._print_summary(data)
        return data

    def _print_summary(self, data: Dict[str, pd.DataFrame]):
        orders = data["orders"]
        print(f"\n  Dataset Summary:")
        print(f"    Orders: {len(orders):,}")
        print(f"    Unique customers: {data['customers']['customer_unique_id'].nunique():,}")
        print(f"    Products: {len(data['products']):,}")
        print(f"    Sellers: {len(data['sellers']):,}")
        date_range = orders["order_purchase_timestamp"].dropna()
        if not date_range.empty:
            print(f"    Date range: {date_range.min()} → {date_range.max()}")
        print(f"    Order statuses: {orders['order_status'].value_counts().to_dict()}\n")


# ============================================================
# STEP 2: DATA CLEANING & ENRICHMENT
# ============================================================

class OlistCleaner:
    """Cleans and enriches Olist data."""

    def __init__(self, config: OlistAdapterConfig):
        self.config = config

    def clean(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run full cleaning pipeline."""
        print("=" * 60)
        print("STEP 2: DATA CLEANING & ENRICHMENT")
        print("=" * 60)

        # 2a. Filter valid order statuses
        orders = data["orders"]
        n_before = len(orders)
        orders = orders[
            orders["order_status"].isin(self.config.valid_order_statuses)
        ].copy()
        print(f"  [2a] Filtered orders: {n_before:,} → {len(orders):,} "
              f"(kept statuses: {list(self.config.valid_order_statuses)})")

        # 2b. Drop orders without purchase timestamp
        orders = orders.dropna(subset=["order_purchase_timestamp"])
        print(f"  [2b] After dropping null timestamps: {len(orders):,}")

        # 2c. Resolve customer_unique_id (same person can have multiple customer_ids)
        # This is critical: Olist assigns different customer_id per order
        customers = data["customers"].copy()
        order_customer = orders.merge(
            customers[["customer_id", "customer_unique_id"]],
            on="customer_id", how="left"
        )
        print(f"  [2c] Resolved {customers['customer_id'].nunique():,} customer_ids "
              f"→ {customers['customer_unique_id'].nunique():,} unique customers")

        # 2d. Filter by minimum orders per unique customer
        user_order_counts = order_customer.groupby("customer_unique_id").size()
        active_users = user_order_counts[
            user_order_counts >= self.config.min_orders_per_user
        ].index
        n_removed = order_customer["customer_unique_id"].nunique() - len(active_users)
        order_customer = order_customer[
            order_customer["customer_unique_id"].isin(active_users)
        ]
        print(f"  [2d] Removed {n_removed:,} users with < {self.config.min_orders_per_user} orders")

        # 2e. Enrich order_items with product and payment info
        order_items = data["order_items"].copy()
        order_items = order_items[order_items["order_id"].isin(orders["order_id"])]

        # Merge product category
        products = data["products"].copy()
        if "category_english" in products.columns:
            order_items = order_items.merge(
                products[["product_id", "category_english",
                          "product_weight_g", "product_length_cm",
                          "product_height_cm", "product_width_cm",
                          "product_photos_qty", "product_name_lenght",
                          "product_description_lenght"]],
                on="product_id", how="left"
            )

        # 2f. Enrich with payment info
        payments = data["payments"].copy()
        payments = payments[payments["order_id"].isin(orders["order_id"])]

        # Aggregate payments per order (an order can have multiple payments)
        payment_agg = payments.groupby("order_id").agg(
            total_payment=("payment_value", "sum"),
            n_installments=("payment_installments", "max"),
            payment_types=("payment_type", lambda x: ",".join(x.unique())),
            n_payment_methods=("payment_type", "nunique"),
        ).reset_index()

        # 2g. Enrich with review scores
        reviews = data["reviews"].copy()
        reviews = reviews[reviews["order_id"].isin(orders["order_id"])]
        review_agg = reviews.groupby("order_id").agg(
            review_score=("review_score", "mean"),
            has_review_comment=("review_comment_message", lambda x: x.notna().any()),
        ).reset_index()

        # 2h. Build enriched orders table
        enriched_orders = order_customer.merge(payment_agg, on="order_id", how="left")
        enriched_orders = enriched_orders.merge(review_agg, on="order_id", how="left")

        # Compute delivery performance
        enriched_orders["delivery_days"] = (
            enriched_orders["order_delivered_customer_date"] -
            enriched_orders["order_purchase_timestamp"]
        ).dt.days

        enriched_orders["delivery_vs_estimate"] = (
            enriched_orders["order_delivered_customer_date"] -
            enriched_orders["order_estimated_delivery_date"]
        ).dt.days  # Negative = early, positive = late

        data["orders"] = enriched_orders
        data["order_items"] = order_items
        data["payments"] = payments
        data["reviews"] = reviews

        print(f"\n  → Clean data: {len(enriched_orders):,} orders | "
              f"{enriched_orders['customer_unique_id'].nunique():,} unique customers | "
              f"{order_items['product_id'].nunique():,} products\n")

        return data


# ============================================================
# STEP 3: CONVERT TO EVOCRM 4-TABLE FORMAT
# ============================================================

class OlistToEvoCRMConverter:
    """
    Converts cleaned Olist data into EvoCRM 4-table schema.

    Olist → EvoCRM mapping:
        customers + geolocation    → demographics
        order_items + payments     → transactions
        order journey events       → web_behavior (synthesized)
        no marketing data          → campaigns (empty)
    """

    def __init__(self, config: OlistAdapterConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def build_demographics(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Build demographics from customers + orders + geolocation.

        ADVANTAGE: Olist has REAL location data (city, state, zip).
        """
        print("  [Table 1/4] Building demographics ...")

        orders = data["orders"]
        customers = data["customers"]

        # Get unique customers with their most recent location
        customer_info = orders.sort_values("order_purchase_timestamp").groupby(
            "customer_unique_id"
        ).agg(
            first_order=("order_purchase_timestamp", "min"),
            last_order=("order_purchase_timestamp", "max"),
            total_orders=("order_id", "nunique"),
            customer_id_last=("customer_id", "last"),
        ).reset_index()

        # Merge location from customers table
        customer_loc = customers[
            ["customer_id", "customer_city", "customer_state", "customer_zip_code_prefix"]
        ].drop_duplicates(subset=["customer_id"])

        customer_info = customer_info.merge(
            customer_loc,
            left_on="customer_id_last", right_on="customer_id",
            how="left"
        )

        # Build demographics table
        demographics = pd.DataFrame({
            "user_id": customer_info["customer_unique_id"],
            "registration_date": customer_info["first_order"],
            "city": customer_info["customer_city"].fillna("unknown"),
            "state": customer_info["customer_state"].fillna("unknown"),
            "zip_prefix": customer_info["customer_zip_code_prefix"].fillna(0).astype(int),
            "tenure_days": (customer_info["last_order"] - customer_info["first_order"]).dt.days,
            "total_orders": customer_info["total_orders"],
            # Derive user segment from order count
            "user_segment": pd.cut(
                customer_info["total_orders"],
                bins=[0, 1, 2, 4, float("inf")],
                labels=["one_time", "returning", "regular", "power_buyer"],
            ).astype(str),
            "gender": "unknown",  # Not available in Olist
        })

        print(f"    → {len(demographics):,} users")
        print(f"    → States: {demographics['state'].nunique()} | "
              f"Cities: {demographics['city'].nunique()}")
        print(f"    → Segments: {demographics['user_segment'].value_counts().to_dict()}")
        return demographics

    def build_transactions(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Build transactions from order_items + orders.

        ADVANTAGE: Olist has REAL prices (no estimation needed).
        """
        print("  [Table 2/4] Building transactions ...")

        orders = data["orders"]
        items = data["order_items"]

        # Merge customer_unique_id and timestamp into order_items
        order_info = orders[[
            "order_id", "customer_unique_id", "order_purchase_timestamp",
            "total_payment", "review_score",
        ]].drop_duplicates(subset=["order_id"])

        txn = items.merge(order_info, on="order_id", how="inner")

        transactions = pd.DataFrame({
            "transaction_id": txn["order_id"] + "_" + txn["order_item_id"].astype(str),
            "user_id": txn["customer_unique_id"],
            "product_id": txn["product_id"],
            "timestamp": txn["order_purchase_timestamp"],
            "amount": txn["price"] + txn["freight_value"],  # Total item cost
            "quantity": 1,  # Olist: each order_item row = 1 unit (quantity encoded in rows)
            # Extra Olist-specific fields
            "price": txn["price"],
            "freight": txn["freight_value"],
            "seller_id": txn["seller_id"],
            "review_score": txn["review_score"],
        })

        print(f"    → {len(transactions):,} transaction items | "
              f"{transactions['user_id'].nunique():,} buyers")
        print(f"    → Avg amount: ${transactions['amount'].mean():.2f} | "
              f"Avg price: ${transactions['price'].mean():.2f} | "
              f"Avg freight: ${transactions['freight'].mean():.2f}")
        return transactions

    def build_web_behavior(
        self, data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Synthesize web_behavior from the order lifecycle.

        Olist has NO browsing data, but each order has a rich lifecycle:
            order_placed → approved → shipped_to_carrier → delivered → reviewed

        We convert each lifecycle stage into an "event", creating a temporal
        sequence that the Interaction Tower can learn from.

        This is documented as "order journey events" in the paper.
        """
        print("  [Table 3/4] Building web_behavior (order journey events) ...")

        orders = data["orders"]
        items = data["order_items"]

        events = []
        event_id_counter = 1

        # For each order, create lifecycle events
        for _, row in orders.iterrows():
            uid = row["customer_unique_id"]
            oid = row["order_id"]

            # Event 1: Order placed
            if pd.notna(row["order_purchase_timestamp"]):
                events.append({
                    "event_id": event_id_counter,
                    "user_id": uid,
                    "product_id": "ORDER_" + str(oid)[:8],
                    "event_type": "order_placed",
                    "timestamp": row["order_purchase_timestamp"],
                    "session_id": oid,
                })
                event_id_counter += 1

            # Event 2: Payment approved
            if pd.notna(row.get("order_approved_at")):
                events.append({
                    "event_id": event_id_counter,
                    "user_id": uid,
                    "product_id": "ORDER_" + str(oid)[:8],
                    "event_type": "payment_approved",
                    "timestamp": row["order_approved_at"],
                    "session_id": oid,
                })
                event_id_counter += 1

            # Event 3: Shipped to carrier
            if pd.notna(row.get("order_delivered_carrier_date")):
                events.append({
                    "event_id": event_id_counter,
                    "user_id": uid,
                    "product_id": "ORDER_" + str(oid)[:8],
                    "event_type": "shipped",
                    "timestamp": row["order_delivered_carrier_date"],
                    "session_id": oid,
                })
                event_id_counter += 1

            # Event 4: Delivered to customer
            if pd.notna(row.get("order_delivered_customer_date")):
                events.append({
                    "event_id": event_id_counter,
                    "user_id": uid,
                    "product_id": "ORDER_" + str(oid)[:8],
                    "event_type": "delivered",
                    "timestamp": row["order_delivered_customer_date"],
                    "session_id": oid,
                })
                event_id_counter += 1

        # Event 5: Add per-item "purchase" events (these map to product_ids)
        order_to_user = orders.set_index("order_id")["customer_unique_id"].to_dict()
        order_to_ts = orders.set_index("order_id")["order_purchase_timestamp"].to_dict()

        for _, row in items.iterrows():
            oid = row["order_id"]
            if oid in order_to_user:
                events.append({
                    "event_id": event_id_counter,
                    "user_id": order_to_user[oid],
                    "product_id": row["product_id"],
                    "event_type": "purchase",
                    "timestamp": order_to_ts.get(oid),
                    "session_id": oid,
                })
                event_id_counter += 1

        # Event 6: Review events
        reviews = data["reviews"]
        order_to_review = reviews.groupby("order_id").first()
        for oid, rev_row in order_to_review.iterrows():
            if oid in order_to_user and pd.notna(rev_row.get("review_creation_date")):
                events.append({
                    "event_id": event_id_counter,
                    "user_id": order_to_user[oid],
                    "product_id": "REVIEW_" + str(oid)[:8],
                    "event_type": f"review_score_{int(rev_row['review_score'])}",
                    "timestamp": rev_row["review_creation_date"],
                    "session_id": oid,
                })
                event_id_counter += 1

        web_behavior = pd.DataFrame(events)
        web_behavior = web_behavior.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
        web_behavior["event_id"] = range(1, len(web_behavior) + 1)

        n_sessions = web_behavior["session_id"].nunique()
        print(f"    → {len(web_behavior):,} order journey events | "
              f"{n_sessions:,} order sessions")
        print(f"    → Event types: {web_behavior['event_type'].value_counts().to_dict()}")
        return web_behavior

    def build_campaigns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Olist has no campaign data — return empty schema-valid table."""
        print("  [Table 4/4] Building campaigns (empty — no marketing data in Olist)")
        campaigns = pd.DataFrame({
            "user_id": pd.Series(dtype="str"),
            "campaign_id": pd.Series(dtype="str"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "clicked": pd.Series(dtype="int64"),
        })
        print("    → 0 campaign records\n")
        return campaigns

    def convert_all(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run the full conversion."""
        print("=" * 60)
        print("STEP 3: CONVERTING TO EVOCRM 4-TABLE FORMAT")
        print("=" * 60)

        return {
            "demographics": self.build_demographics(data),
            "transactions": self.build_transactions(data),
            "web_behavior": self.build_web_behavior(data),
            "campaigns": self.build_campaigns(data),
        }


# ============================================================
# STEP 4: GENERATE TARGET VARIABLES
# ============================================================

class OlistTargetGenerator:
    """Generates all 6 EvoCRM task head targets from Olist data."""

    def __init__(self, config: OlistAdapterConfig):
        self.config = config

    def generate_all(
        self,
        transactions: pd.DataFrame,
        web_behavior: pd.DataFrame,
        demographics: pd.DataFrame,
        raw_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Generate all 6 targets."""
        print("=" * 60)
        print("STEP 4: GENERATING TARGET VARIABLES (6 TASK HEADS)")
        print("=" * 60)

        users = demographics[["user_id"]].copy()
        data_end = transactions["timestamp"].max()

        users = self._churn(users, transactions, data_end)
        users = self._clv(users, transactions)
        users = self._upsell(users, transactions)
        users = self._next_item(users, transactions)
        users = self._early_adopter(users, transactions)
        users = self._days_next_purchase(users, transactions, data_end)

        # --- BONUS: Olist-specific targets ---
        users = self._satisfaction_risk(users, transactions, raw_data)

        self._print_summary(users)
        return users

    def _churn(self, users, txn, data_end):
        """Churn = No order in the last N days."""
        print("  [Target 1/6] Churn Prediction ...")

        cutoff = data_end - pd.Timedelta(days=self.config.churn_window_days)

        if txn.empty:
            users["churn"] = 1
        else:
            last_order = txn.groupby("user_id")["timestamp"].max().reset_index()
            last_order.columns = ["user_id", "last_order_ts"]
            users = users.merge(last_order, on="user_id", how="left")
            users["churn"] = (
                users["last_order_ts"].isna() | (users["last_order_ts"] < cutoff)
            ).astype(int)
            users = users.drop(columns=["last_order_ts"])

        print(f"    → Churn rate: {users['churn'].mean():.1%}")
        return users

    def _clv(self, users, txn):
        """CLV = Total spend."""
        print("  [Target 2/6] CLV Estimation ...")

        if txn.empty:
            users["clv"] = 0.0
        else:
            clv = txn.groupby("user_id")["amount"].sum().reset_index()
            clv.columns = ["user_id", "clv"]
            users = users.merge(clv, on="user_id", how="left")
            users["clv"] = users["clv"].fillna(0.0)

        print(f"    → CLV: mean=${users['clv'].mean():.2f}, median=${users['clv'].median():.2f}")
        return users

    def _upsell(self, users, txn):
        """Upsell = AOV increased in second half of history."""
        print("  [Target 3/6] Upsell Detection ...")

        if txn.empty:
            users["upsell"] = 0
            return users

        # Group by user+order to get order-level AOV
        order_aov = txn.groupby(["user_id", "timestamp"]).agg(
            order_total=("amount", "sum")
        ).reset_index()
        order_aov = order_aov.sort_values(["user_id", "timestamp"])

        # For users with 2+ orders, compare first vs second half
        def compute_upsell(group):
            if len(group) < 2:
                return 0
            mid = len(group) // 2
            first_aov = group.iloc[:mid]["order_total"].mean()
            second_aov = group.iloc[mid:]["order_total"].mean()
            if first_aov <= 0:
                return 0
            return int((second_aov - first_aov) / first_aov >= self.config.upsell_aov_increase_pct)

        upsell = order_aov.groupby("user_id").apply(compute_upsell).reset_index()
        upsell.columns = ["user_id", "upsell"]
        users = users.merge(upsell, on="user_id", how="left")
        users["upsell"] = users["upsell"].fillna(0).astype(int)

        print(f"    → Upsell rate: {users['upsell'].mean():.1%}")
        return users

    def _next_item(self, users, txn):
        """Next item = Last purchased product_id."""
        print("  [Target 4/6] Next-Item Recommendation ...")

        if txn.empty:
            users["next_item_id"] = "NONE"
        else:
            last = txn.sort_values("timestamp").groupby("user_id")["product_id"].last()
            last = last.reset_index()
            last.columns = ["user_id", "next_item_id"]
            users = users.merge(last, on="user_id", how="left")
            users["next_item_id"] = users["next_item_id"].fillna("NONE")

        valid = (users["next_item_id"] != "NONE").sum()
        print(f"    → {valid:,} users with valid next-item target")
        return users

    def _early_adopter(self, users, txn):
        """Early adopter = Bought item within N days of its first appearance."""
        print("  [Target 5/6] Early Adopter Scoring ...")

        if txn.empty:
            users["early_adopter"] = 0
            return users

        item_first = txn.groupby("product_id")["timestamp"].min().reset_index()
        item_first.columns = ["product_id", "first_seen"]

        txn_merged = txn.merge(item_first, on="product_id")
        txn_merged["days_since_launch"] = (
            txn_merged["timestamp"] - txn_merged["first_seen"]
        ).dt.days

        early_users = txn_merged[
            txn_merged["days_since_launch"] <= self.config.early_adopter_days
        ]["user_id"].unique()

        users["early_adopter"] = users["user_id"].isin(early_users).astype(int)
        print(f"    → Early adopter rate: {users['early_adopter'].mean():.1%}")
        return users

    def _days_next_purchase(self, users, txn, data_end):
        """Days between last two orders."""
        print("  [Target 6/6] Days Until Next Purchase ...")

        if txn.empty:
            users["days_next_purchase"] = self.config.next_purchase_max_days
            return users

        # Get per-user order timestamps (deduplicated by date)
        order_dates = txn.groupby(["user_id", txn["timestamp"].dt.date]).first().reset_index(level=1)

        def calc_gap(group):
            dates = sorted(group["timestamp"].dropna().tolist())
            if len(dates) >= 2:
                gap = (dates[-1] - dates[-2]).days
                return min(gap, self.config.next_purchase_max_days)
            elif len(dates) == 1:
                return min((data_end - dates[0]).days, self.config.next_purchase_max_days)
            return self.config.next_purchase_max_days

        gaps = txn.groupby("user_id").apply(calc_gap).reset_index()
        gaps.columns = ["user_id", "days_next_purchase"]

        users = users.merge(gaps, on="user_id", how="left")
        users["days_next_purchase"] = users["days_next_purchase"].fillna(
            self.config.next_purchase_max_days
        )

        print(f"    → Avg days: {users['days_next_purchase'].mean():.1f}")
        return users

    def _satisfaction_risk(self, users, txn, raw_data):
        """
        BONUS Olist-specific target: Satisfaction risk from review scores.
        This is unique to Olist and can be an additional task head.
        """
        print("  [Bonus Target] Satisfaction Risk (Olist-specific) ...")

        if "review_score" in txn.columns:
            avg_score = txn.groupby("user_id")["review_score"].mean().reset_index()
            avg_score.columns = ["user_id", "avg_review_score"]
            users = users.merge(avg_score, on="user_id", how="left")
            users["avg_review_score"] = users["avg_review_score"].fillna(3.0)

            # Binary: satisfaction risk (low review score)
            users["satisfaction_risk"] = (
                users["avg_review_score"] <= self.config.dissatisfied_threshold
            ).astype(int)

            print(f"    → Avg review: {users['avg_review_score'].mean():.2f} | "
                  f"At-risk rate: {users['satisfaction_risk'].mean():.1%}")
        else:
            users["avg_review_score"] = 3.0
            users["satisfaction_risk"] = 0

        return users

    def _print_summary(self, users):
        print(f"\n  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │              TARGET VARIABLE SUMMARY                    │")
        print(f"  ├───────────────────────────┬─────────────────────────────┤")
        print(f"  │ Churn (binary)            │ rate = {users['churn'].mean():.1%}"
              f"{'':>18}│")
        print(f"  │ CLV (continuous)          │ mean = ${users['clv'].mean():.2f}"
              f"{'':>14}│")
        print(f"  │ Upsell (binary)           │ rate = {users['upsell'].mean():.1%}"
              f"{'':>18}│")
        print(f"  │ Next Item (categorical)   │ {(users['next_item_id'] != 'NONE').sum():,} valid"
              f"{'':>15}│")
        print(f"  │ Early Adopter (binary)    │ rate = {users['early_adopter'].mean():.1%}"
              f"{'':>18}│")
        print(f"  │ Days Next Purch (cont.)   │ mean = {users['days_next_purchase'].mean():.1f} days"
              f"{'':>10}│")
        print(f"  │ Satisfaction Risk (bonus)  │ rate = {users['satisfaction_risk'].mean():.1%}"
              f"{'':>18}│")
        print(f"  └───────────────────────────┴─────────────────────────────┘")


# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

class OlistFeatureEngineer:
    """
    Builds tower-specific features from Olist data.

    Key advantage over RetailRocket:
        - Real monetary features (price, freight, payment)
        - Real geographic features (city, state, zip)
        - Real satisfaction signals (review scores)
        - Delivery performance features
    """

    def __init__(self, config: OlistAdapterConfig):
        self.config = config

    def build_customer_tower_features(
        self,
        demographics: pd.DataFrame,
        transactions: pd.DataFrame,
        web_behavior: pd.DataFrame,
        raw_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Build rich customer features using all Olist signals."""
        print("  [Tower 1/3] Customer Tower Features ...")

        users = demographics[["user_id", "city", "state", "user_segment", "tenure_days"]].copy()
        data_end = transactions["timestamp"].max()

        # === RFM Features (REAL prices) ===
        if not transactions.empty:
            rfm = transactions.groupby("user_id").agg(
                recency_days=("timestamp", lambda x: (data_end - x.max()).days),
                frequency=("transaction_id", "nunique"),
                monetary=("amount", "sum"),
                avg_order_value=("amount", "mean"),
                max_order_value=("amount", "max"),
                min_order_value=("amount", "min"),
                std_order_value=("amount", "std"),
                total_items=("transaction_id", "count"),
                n_unique_products=("product_id", "nunique"),
            ).reset_index()
            rfm["std_order_value"] = rfm["std_order_value"].fillna(0)
            users = users.merge(rfm, on="user_id", how="left")

        # === Freight & Pricing Features ===
        if "price" in transactions.columns:
            price_stats = transactions.groupby("user_id").agg(
                avg_item_price=("price", "mean"),
                total_freight=("freight", "sum"),
                avg_freight=("freight", "mean"),
                freight_ratio=("freight", lambda x: x.sum()),  # Will compute ratio below
            ).reset_index()
            users = users.merge(price_stats, on="user_id", how="left")
            # Freight as % of total spend
            if "monetary" in users.columns:
                users["freight_pct"] = (
                    users["total_freight"] / users["monetary"].clip(lower=0.01)
                )

        # === Review / Satisfaction Features ===
        if "review_score" in transactions.columns:
            review_stats = transactions.groupby("user_id").agg(
                avg_review_score=("review_score", "mean"),
                min_review_score=("review_score", "min"),
                max_review_score=("review_score", "max"),
                n_reviews=("review_score", lambda x: x.notna().sum()),
            ).reset_index()
            users = users.merge(review_stats, on="user_id", how="left")

        # === Seller Diversity Features ===
        if "seller_id" in transactions.columns:
            seller_stats = transactions.groupby("user_id").agg(
                n_unique_sellers=("seller_id", "nunique"),
            ).reset_index()
            users = users.merge(seller_stats, on="user_id", how="left")

        # === Delivery Performance Features ===
        orders = raw_data.get("orders", pd.DataFrame())
        if not orders.empty and "delivery_days" in orders.columns:
            delivery_stats = orders.groupby("customer_unique_id").agg(
                avg_delivery_days=("delivery_days", "mean"),
                max_delivery_days=("delivery_days", "max"),
                n_late_deliveries=("delivery_vs_estimate",
                                   lambda x: (x > 0).sum()),
                avg_delivery_delta=("delivery_vs_estimate", "mean"),
            ).reset_index()
            delivery_stats = delivery_stats.rename(
                columns={"customer_unique_id": "user_id"}
            )
            users = users.merge(delivery_stats, on="user_id", how="left")

        # === Payment Features ===
        payments = raw_data.get("payments", pd.DataFrame())
        if not payments.empty:
            # Map orders to users
            order_user = orders[["order_id", "customer_unique_id"]].drop_duplicates()
            pay_with_user = payments.merge(order_user, on="order_id", how="inner")
            pay_stats = pay_with_user.groupby("customer_unique_id").agg(
                avg_installments=("payment_installments", "mean"),
                max_installments=("payment_installments", "max"),
                n_payment_types=("payment_type", "nunique"),
                uses_credit_card=("payment_type",
                                  lambda x: int("credit_card" in x.values)),
                uses_boleto=("payment_type",
                             lambda x: int("boleto" in x.values)),
            ).reset_index()
            pay_stats = pay_stats.rename(
                columns={"customer_unique_id": "user_id"}
            )
            users = users.merge(pay_stats, on="user_id", how="left")

        # === Category Preference Features ===
        if "category_english" in transactions.columns or "category_english" in raw_data.get("order_items", pd.DataFrame()).columns:
            items = raw_data.get("order_items", pd.DataFrame())
            if "category_english" in items.columns:
                order_user_map = orders.set_index("order_id")["customer_unique_id"].to_dict()
                items["user_id"] = items["order_id"].map(order_user_map)
                cat_stats = items.dropna(subset=["user_id"]).groupby("user_id").agg(
                    n_unique_categories=("category_english", "nunique"),
                    top_category=("category_english",
                                  lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"),
                ).reset_index()
                users = users.merge(cat_stats, on="user_id", how="left")

        # === Order Journey Features (from web_behavior) ===
        if not web_behavior.empty:
            journey_stats = web_behavior.groupby("user_id").agg(
                total_journey_events=("event_id", "count"),
                n_order_sessions=("session_id", "nunique"),
            ).reset_index()
            users = users.merge(journey_stats, on="user_id", how="left")

        # Fill NaN
        numeric_cols = users.select_dtypes(include=[np.number]).columns
        users[numeric_cols] = users[numeric_cols].fillna(0)
        string_cols = users.select_dtypes(include=["object"]).columns
        for col in string_cols:
            users[col] = users[col].fillna("unknown")

        print(f"    → {len(users):,} users × {len(users.columns)} features")
        return users

    def build_interaction_sequences(
        self,
        web_behavior: pd.DataFrame,
        transactions: pd.DataFrame,
        max_seq_length: int = 128,
    ) -> Dict[str, List[List]]:
        """
        Build per-user event sequences for the Interaction Tower.

        For Olist, the sequence is the order journey:
            order_placed → approved → shipped → delivered → reviewed + product purchases
        """
        print("  [Tower 2/3] Interaction Tower Sequences ...")

        EVENT_TYPE_IDS = {
            "order_placed": 1, "payment_approved": 2, "shipped": 3,
            "delivered": 4, "purchase": 5,
            "review_score_1": 6, "review_score_2": 7, "review_score_3": 8,
            "review_score_4": 9, "review_score_5": 10,
        }

        web = web_behavior.sort_values(["user_id", "timestamp"]).copy()
        web["event_type_id"] = web["event_type"].map(EVENT_TYPE_IDS).fillna(0).astype(int)
        web["first_ts"] = web.groupby("user_id")["timestamp"].transform("min")
        web["time_delta"] = (web["timestamp"] - web["first_ts"]).dt.total_seconds().fillna(0)

        # Build product_id numeric mapping
        all_products = web["product_id"].unique()
        product_to_idx = {pid: idx + 1 for idx, pid in enumerate(all_products)}

        sequences = {}
        for user_id, group in web.groupby("user_id"):
            events = []
            for _, row in group.iterrows():
                events.append([
                    row["event_type_id"],
                    product_to_idx.get(row["product_id"], 0),
                    row["time_delta"],
                ])
            # Truncate
            if len(events) > max_seq_length:
                events = events[-max_seq_length:]
            sequences[str(user_id)] = events

        lengths = [len(v) for v in sequences.values()]
        print(f"    → {len(sequences):,} user sequences")
        print(f"    → Avg length: {np.mean(lengths):.1f} | "
              f"Median: {np.median(lengths):.0f} | "
              f"Event types: {len(EVENT_TYPE_IDS)}")
        return sequences

    def build_product_features(
        self, raw_data: Dict[str, pd.DataFrame], transactions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build Product Tower features.

        ADVANTAGE: Olist has real product attributes (weight, dimensions,
        photos count, description length, category).
        """
        print("  [Tower 3/3] Product Tower Features ...")

        products = raw_data.get("products", pd.DataFrame()).copy()
        items = raw_data.get("order_items", pd.DataFrame())

        if products.empty:
            print("    → WARNING: No product data available")
            return pd.DataFrame(columns=["product_id"])

        # Sales statistics
        if not items.empty:
            sales_stats = items.groupby("product_id").agg(
                n_orders=("order_id", "nunique"),
                total_revenue=("price", "sum"),
                avg_price=("price", "mean"),
                avg_freight=("freight_value", "mean"),
                n_sellers=("seller_id", "nunique"),
            ).reset_index()
            products = products.merge(sales_stats, on="product_id", how="left")

        # Review stats per product
        if not transactions.empty and "review_score" in transactions.columns:
            prod_reviews = transactions.groupby("product_id").agg(
                product_avg_review=("review_score", "mean"),
                product_n_reviews=("review_score", lambda x: x.notna().sum()),
            ).reset_index()
            products = products.merge(prod_reviews, on="product_id", how="left")

        # Fill NaN
        fill_cols = [
            "product_weight_g", "product_length_cm", "product_height_cm",
            "product_width_cm", "product_photos_qty", "product_name_lenght",
            "product_description_lenght", "n_orders", "total_revenue",
            "avg_price", "avg_freight", "n_sellers",
            "product_avg_review", "product_n_reviews",
        ]
        for col in fill_cols:
            if col in products.columns:
                products[col] = products[col].fillna(0)

        # Compute volume
        for col in ["product_length_cm", "product_height_cm", "product_width_cm"]:
            if col in products.columns:
                products[col] = products[col].clip(lower=0)
        if all(c in products.columns for c in ["product_length_cm", "product_height_cm", "product_width_cm"]):
            products["product_volume_cm3"] = (
                products["product_length_cm"] *
                products["product_height_cm"] *
                products["product_width_cm"]
            )

        # Category encoding
        if "category_english" in products.columns:
            products["category_english"] = products["category_english"].fillna("unknown")
        if "product_category_name" in products.columns:
            products["product_category_name"] = products["product_category_name"].fillna("unknown")

        print(f"    → {len(products):,} products × {len(products.columns)} features")
        print(f"    → Has physical attributes: weight, dimensions, photos ✓")
        print(f"    → Has category names (for Sentence-BERT if desired) ✓")
        return products


# ============================================================
# STEP 6: DATA EXPORT
# ============================================================

class OlistExporter:
    """Exports processed data in EvoCRM-ready format."""

    def __init__(self, config: OlistAdapterConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def export(
        self,
        tables: Dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        customer_features: pd.DataFrame,
        product_features: pd.DataFrame,
        interaction_sequences: Dict,
        metadata: Dict,
    ) -> Path:
        """Save everything to disk."""
        output = Path(self.config.output_dir)
        output.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("STEP 6: EXPORTING EVOCRM-READY DATA")
        print("=" * 60)

        # Tables
        tables_dir = output / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, df in tables.items():
            df.to_csv(tables_dir / f"{name}.csv", index=False)
            print(f"  ✓ {name}.csv ({len(df):,} rows)")

        # Targets
        targets.to_csv(output / "targets.csv", index=False)
        print(f"  ✓ targets.csv ({len(targets):,} rows)")

        # Features
        feat_dir = output / "features"
        feat_dir.mkdir(exist_ok=True)
        customer_features.to_csv(feat_dir / "customer_tower_features.csv", index=False)
        print(f"  ✓ customer_tower_features.csv ({len(customer_features):,} rows)")
        product_features.to_csv(feat_dir / "product_tower_features.csv", index=False)
        print(f"  ✓ product_tower_features.csv ({len(product_features):,} rows)")

        # Sequences
        seq_path = feat_dir / "interaction_sequences.json"
        with open(seq_path, "w") as f:
            json.dump(interaction_sequences, f)
        print(f"  ✓ interaction_sequences.json ({len(interaction_sequences):,} users)")

        # Splits
        user_ids = targets["user_id"].values
        n = len(user_ids)
        shuffled = self.rng.permutation(user_ids)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        splits = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }
        splits_dir = output / "splits"
        splits_dir.mkdir(exist_ok=True)
        for name, ids in splits.items():
            np.save(splits_dir / f"{name}_user_ids.npy", ids)
            print(f"  ✓ {name}_user_ids.npy ({len(ids):,})")

        # Metadata
        with open(output / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  ✓ metadata.json")

        print(f"\n  All files exported to: {output.resolve()}")
        return output


# ============================================================
# STEP 7: MAIN PIPELINE
# ============================================================

class OlistToEvoCRM:
    """Main pipeline orchestrator."""

    def __init__(self, config: OlistAdapterConfig):
        self.config = config

    def run(self) -> Path:
        """Execute full pipeline."""
        print("╔" + "═" * 58 + "╗")
        print("║   Olist Brazilian E-Commerce → EvoCRM Adapter          ║")
        print("║   Real data: prices, locations, reviews, delivery      ║")
        print("╚" + "═" * 58 + "╝\n")

        # Step 1: Load
        loader = OlistLoader(self.config)
        raw = loader.load_all()

        # Step 2: Clean
        cleaner = OlistCleaner(self.config)
        data = cleaner.clean(raw)

        # Step 3: Convert to 4 tables
        converter = OlistToEvoCRMConverter(self.config)
        tables = converter.convert_all(data)

        # Step 4: Generate targets
        target_gen = OlistTargetGenerator(self.config)
        targets = target_gen.generate_all(
            transactions=tables["transactions"],
            web_behavior=tables["web_behavior"],
            demographics=tables["demographics"],
            raw_data=data,
        )

        # Step 5: Feature engineering
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE ENGINEERING FOR EVOCRM TOWERS")
        print("=" * 60)

        feat_eng = OlistFeatureEngineer(self.config)

        customer_features = feat_eng.build_customer_tower_features(
            demographics=tables["demographics"],
            transactions=tables["transactions"],
            web_behavior=tables["web_behavior"],
            raw_data=data,
        )

        interaction_sequences = feat_eng.build_interaction_sequences(
            web_behavior=tables["web_behavior"],
            transactions=tables["transactions"],
            max_seq_length=128,
        )

        product_features = feat_eng.build_product_features(
            raw_data=data,
            transactions=tables["transactions"],
        )

        # Build metadata
        cat_cols = ["city", "state", "user_segment"]
        extra_cat = [c for c in ["top_category"] if c in customer_features.columns]
        cat_cols += extra_cat
        num_cols = [c for c in customer_features.select_dtypes(include=[np.number]).columns
                    if c != "user_id"]

        metadata = {
            "source_dataset": "Olist Brazilian E-Commerce",
            "source_url": "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce",
            "conversion_date": datetime.now().isoformat(),
            "config": {
                "min_orders_per_user": self.config.min_orders_per_user,
                "churn_window_days": self.config.churn_window_days,
                "churn_gap_days": self.config.churn_gap_days,
                "valid_order_statuses": list(self.config.valid_order_statuses),
                "seed": self.config.seed,
            },
            "stats": {
                "n_users": len(tables["demographics"]),
                "n_transactions": len(tables["transactions"]),
                "n_web_events": len(tables["web_behavior"]),
                "n_products": product_features["product_id"].nunique() if "product_id" in product_features.columns else 0,
                "n_categories": product_features["category_english"].nunique() if "category_english" in product_features.columns else 0,
            },
            "customer_tower": {
                "categorical_features": cat_cols,
                "categorical_cardinalities": {
                    col: int(customer_features[col].nunique())
                    for col in cat_cols if col in customer_features.columns
                },
                "numerical_features": num_cols,
                "num_numerical": len(num_cols),
                "num_categorical": len(cat_cols),
            },
            "interaction_tower": {
                "max_sequence_length": 128,
                "num_event_types": 10,
                "event_type_map": {
                    "order_placed": 1, "payment_approved": 2, "shipped": 3,
                    "delivered": 4, "purchase": 5,
                    "review_1": 6, "review_2": 7, "review_3": 8,
                    "review_4": 9, "review_5": 10,
                },
                "note": "Events are order journey stages, not web browsing",
            },
            "product_tower": {
                "num_products": int(product_features["product_id"].nunique()) if "product_id" in product_features.columns else 0,
                "has_text_descriptions": True,
                "has_physical_attributes": True,
                "note": "Real categories available for Sentence-BERT; physical attributes (weight, dimensions) also available",
            },
            "targets": {
                "churn": {"type": "binary", "positive_rate": float(targets["churn"].mean())},
                "clv": {"type": "continuous", "mean": float(targets["clv"].mean())},
                "upsell": {"type": "binary", "positive_rate": float(targets["upsell"].mean())},
                "next_item": {"type": "categorical", "n_valid": int((targets["next_item_id"] != "NONE").sum())},
                "early_adopter": {"type": "binary", "positive_rate": float(targets["early_adopter"].mean())},
                "days_next_purchase": {"type": "continuous", "mean": float(targets["days_next_purchase"].mean())},
                "satisfaction_risk": {"type": "binary", "positive_rate": float(targets["satisfaction_risk"].mean()), "note": "Olist bonus target"},
            },
            "splits": {
                "train_ratio": self.config.train_ratio,
                "val_ratio": self.config.val_ratio,
                "test_ratio": self.config.test_ratio,
            },
            "advantages_over_retailrocket": [
                "Real prices and payment amounts (no estimation)",
                "Real customer locations (city, state, zip)",
                "Real product categories with English translations",
                "Review scores (1-5) as satisfaction signals",
                "Delivery performance data (early/late delivery)",
                "Payment method diversity (credit card, boleto, voucher, debit)",
                "Physical product attributes (weight, dimensions, photos)",
                "Product text fields available for Sentence-BERT",
            ],
            "limitations": [
                "No web browsing data — Interaction Tower uses order journey events",
                "No campaign/marketing data — Campaign Tower zeroed",
                "~96K unique customers — moderate size",
                "Marketplace model (multi-seller) differs from single-brand DTC",
                "Most customers have only 1 order — limits sequence modeling",
                "Brazilian market only — may not generalize globally",
            ],
        }

        # Step 6: Export
        exporter = OlistExporter(self.config)
        output_dir = exporter.export(
            tables=tables,
            targets=targets,
            customer_features=customer_features,
            product_features=product_features,
            interaction_sequences=interaction_sequences,
            metadata=metadata,
        )

        self._print_final_summary(metadata, targets)
        return output_dir

    def _print_final_summary(self, meta, targets):
        """Print comprehensive pipeline summary."""
        print("\n" + "╔" + "═" * 58 + "╗")
        print("║            OLIST PIPELINE COMPLETE                      ║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Source: Olist Brazilian E-Commerce (Kaggle)            ║")
        print(f"║  Users:        {meta['stats']['n_users']:>8,}                              ║")
        print(f"║  Transactions: {meta['stats']['n_transactions']:>8,}                              ║")
        print(f"║  Journey Evts: {meta['stats']['n_web_events']:>8,}                              ║")
        print(f"║  Products:     {meta['stats']['n_products']:>8,}                              ║")
        print("╠" + "═" * 58 + "╣")
        print("║  REAL DATA ADVANTAGES (vs RetailRocket):                ║")
        print("║    ✓ Real prices  ✓ Real locations  ✓ Review scores     ║")
        print("║    ✓ Delivery perf ✓ Payment types  ✓ Product attrs    ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Targets: Churn | CLV | Upsell | RecSys | Early Adopter║")
        print("║           Days Next Purchase | Satisfaction Risk (bonus)║")
        print("╠" + "═" * 58 + "╣")
        print("║  LIMITATIONS (document in paper):                       ║")
        print("║    ✗ No web browsing (order journey instead)            ║")
        print("║    ✗ No campaigns  ✗ Most users = 1 order only         ║")
        print("╚" + "═" * 58 + "╝")

        print("\n  NEXT STEPS:")
        print("  1. python validate_adapter_output.py --data_dir ./evocrm_olist/")
        print("  2. python train_baselines.py --data ./evocrm_olist/")
        print("  3. python train_evocrm.py --phase towers --data ./evocrm_olist/")
        print("  4. Compare results against RetailRocket for cross-dataset validation\n")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert Olist Brazilian E-Commerce dataset to EvoCRM format"
    )
    parser.add_argument("--data_dir", type=str, default="./olist_raw/")
    parser.add_argument("--output_dir", type=str, default="./evocrm_olist/")
    parser.add_argument("--min_orders", type=int, default=1)
    parser.add_argument("--churn_window", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    config = OlistAdapterConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_orders_per_user=args.min_orders,
        churn_window_days=args.churn_window,
        seed=args.seed,
    )

    pipeline = OlistToEvoCRM(config)
    pipeline.run()


if __name__ == "__main__":
    main()
