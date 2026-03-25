# Olist Brazilian E-Commerce → EvoCRM Data Adapter

Converts the [Olist public dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (8 relational tables) into EvoCRM format with all 6+1 task head targets.

## Why Olist Is Stronger Than RetailRocket for Your Paper

| Feature | RetailRocket | Olist |
|---|---|---|
| Prices | ✗ Estimated | ✓ **Real** |
| Customer location | ✗ Hashed | ✓ **City, state, zip** |
| Product categories | ✗ Hashed | ✓ **English names** |
| Review scores | ✗ None | ✓ **1–5 scale** |
| Delivery performance | ✗ None | ✓ **Estimated vs actual** |
| Payment methods | ✗ None | ✓ **Credit, boleto, voucher** |
| Product dimensions | ✗ None | ✓ **Weight, length, height, width** |
| Sentence-BERT usable | ✗ No (hashed) | ✓ **Yes (category names)** |
| Web browsing data | ✓ Views/carts | ✗ Order journey only |
| Dataset size | ~1.4M visitors | ~96K customers |

## Quick Start

```bash
# 1. Download from Kaggle
kaggle datasets download -d olistbr/brazilian-ecommerce
unzip brazilian-ecommerce.zip -d olist_raw/

# 2. Run adapter
python olist_adapter.py --data_dir ./olist_raw/ --output_dir ./evocrm_olist/

# 3. Validate
python validate_adapter_output.py --data_dir ./evocrm_olist/

# 4. Train
python train_evocrm.py --data ./evocrm_olist/ --phase towers
```

## Olist Source Tables → EvoCRM Mapping

```
Olist (8 tables)                         EvoCRM Format
─────────────────                        ──────────────
olist_customers_dataset.csv    ──┐
olist_geolocation_dataset.csv  ──┤─→     demographics.csv (REAL locations)
                                 │
olist_orders_dataset.csv       ──┤─→     transactions.csv (REAL prices)
olist_order_items_dataset.csv  ──┤
olist_order_payments_dataset.csv─┤
                                 │
  (order lifecycle events)     ──┤─→     web_behavior.csv (order journey)
olist_order_reviews_dataset.csv──┤
                                 │
olist_products_dataset.csv     ──┤─→     product_tower_features.csv
category_translation.csv       ──┘       (REAL attributes + categories)
```

## Interaction Tower: Order Journey Sequences

Since Olist has no web browsing data, we model the **order lifecycle** as a temporal sequence:

```
User Timeline:
  ├─ order_placed (t=0)
  ├─ payment_approved (t=2h)
  ├─ purchase: product_A (t=2h)
  ├─ purchase: product_B (t=2h)
  ├─ shipped (t=3 days)
  ├─ delivered (t=12 days)
  ├─ review_score_4 (t=15 days)
  ├─ order_placed (t=45 days)     ← next order
  └─ ...
```

Each event → (event_type_id, product_id_mapped, time_delta_seconds)

## Customer Tower: 35+ Real Features

Olist provides much richer customer features than RetailRocket:

- **RFM:** recency, frequency, monetary (real BRL amounts)
- **Pricing:** avg item price, freight costs, freight % of total
- **Reviews:** avg score, min/max score, review count
- **Delivery:** avg delivery days, late delivery count, delivery delta
- **Payments:** installments, credit card usage, boleto usage
- **Categories:** number of unique categories, top category
- **Geography:** city, state, zip prefix (real Brazilian locations)

## Targets Generated (6 + 1 Bonus)

| Target | Type | Notes |
|---|---|---|
| Churn | Binary | No order in last 90 days |
| CLV | Continuous | Total spend in BRL (real) |
| Upsell | Binary | 20%+ AOV increase |
| Next Item | Categorical | Last purchased product_id |
| Early Adopter | Binary | Bought within 30 days of item launch |
| Days Next Purchase | Continuous | Gap between last two orders |
| **Satisfaction Risk** | **Binary (bonus)** | **Avg review ≤ 2 stars** |

## Known Limitations

1. **No web browsing** — Interaction Tower uses order journey, not page views
2. **No campaigns** — Campaign tower zeroed
3. **Most users have 1 order** — Limits temporal sequence depth
4. **Marketplace model** — Multi-seller, not single-brand DTC
5. **Brazilian market** — May not generalize globally

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
```
