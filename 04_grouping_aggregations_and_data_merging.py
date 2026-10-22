"""
================================================================================
Day 25: Python Data Analysis - Grouping, Aggregations & Relational Merges
================================================================================
The ability to segment datasets, compute multi-dimensional aggregations, and
merge disparate relational data sources is the cornerstone of business intelligence,
analytics engineering, and machine learning feature extraction.

Theoretical Background:
The "Split-Apply-Combine" paradigm (formalized by Hadley Wickham, 2011):
  1. Split: Partition a tabular dataset into disjoint subsets based on keys.
  2. Apply: Compute a reduction (aggregation), transformation, or filtration
     independently on each partition.
  3. Combine: Stitch the individual partition results back into an aligned data structure.

Key Topics Covered in this Script:
1. The Split-Apply-Combine Paradigm & GroupBy Mechanics
2. Modern Named Aggregations & Custom Statistical Reductions
3. Group-Level Window Functions & Transformations with `.transform()`
4. Group Filtering via `.filter()` (Cohort Pruning)
5. Relational Joins: Inner, Left, Right, Full Outer & Cross Joins (`pd.merge`)
6. Merge Provenance Auditing (`indicator=True`) & Schema Validation (`validate=...`)
7. Multidimensional Concatenation: Stacking & Side-by-Side Merging (`pd.concat`)
8. Time Series Windowing: Resampling, Rolling Statistics & Lag Features (`shift`)
9. Production Case Study: Omnichannel Retail Sales Performance & Ranking Engine
================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


def print_separator(title: str) -> None:
    """Helper to print formatted section banners for clean console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. THE SPLIT-APPLY-COMBINE PARADIGM & GROUPBY INTERNALS
# ==============================================================================
# When calling `df.groupby('dept')`, Pandas creates a lazy GroupBy object.
# It computes an internal hash-based index mapping each key to its row locations.

print_separator("1. GroupBy Internals & Group Inspection")

transactions = pd.DataFrame({
    "store_id": ["S1", "S1", "S2", "S1", "S2", "S3", "S3", "S2"],
    "category": ["Electronics", "Clothing", "Electronics", "Clothing", "Home", "Electronics", "Clothing", "Home"],
    "revenue": [1200.0, 350.0, 950.0, 420.0, 210.0, 1850.0, 600.0, 180.0],
    "units": [4, 7, 3, 9, 2, 6, 12, 1]
})

print("Sample Retail Transactions:")
print(transactions)

# Lazy GroupBy object inspection
grouped = transactions.groupby("store_id")
print(f"\nGroup Keys identified: {list(grouped.groups.keys())}")
print(f"Row indices mapped to Store S1: {grouped.groups['S1'].tolist()}")

# Iterating over groups (useful for custom sub-file exports or model training)
print("\nIterating over first 2 groups:")
for store, group_df in list(grouped)[:2]:
    print(f"  Store '{store}' total sub-records: {len(group_df)}, Revenue: ${group_df['revenue'].sum():,.2f}")


# ==============================================================================
# 2. MODERN NAMED AGGREGATIONS & CUSTOM STATISTICAL REDUCTIONS
# ==============================================================================
# Pandas 0.25+ introduced "Named Aggregations" syntax:
# `new_col_name = pd.NamedAgg(column='col', aggfunc='func')` or `new_col=('col', 'func')`
# This avoids ugly multi-level index column names!

print_separator("2. Modern Named Aggregations & Custom Metrics")

# Custom metric: Coefficient of Variation (std / mean)
def coeff_of_variation(series: pd.Series) -> float:
    mean = series.mean()
    return (series.std() / mean) if mean != 0 else 0.0

category_metrics = transactions.groupby("category").agg(
    transaction_count=("revenue", "count"),
    total_revenue=("revenue", "sum"),
    average_order_value=("revenue", "mean"),
    max_single_sale=("revenue", "max"),
    total_units_sold=("units", "sum"),
    revenue_cv=("revenue", coeff_of_variation)
)

print("Named Aggregations by Product Category:")
print(category_metrics.round(2))


# ==============================================================================
# 3. GROUP-LEVEL TRANSFORMATIONS VIA .transform()
# ==============================================================================
# Unlike `.agg()` which collapses rows, `.transform()` produces an output Series
# having the EXACT SAME NUMBER OF ROWS as the original DataFrame.
# Crucial for calculating percentages of group totals, z-scores, and group deviations.

print_separator("3. Group-Level Transformations (.transform)")

# 1. Calculate store total revenue aligned to each individual transaction
transactions["store_total_revenue"] = transactions.groupby("store_id")["revenue"].transform("sum")

# 2. Calculate percentage contribution of this transaction to its store's revenue
transactions["pct_of_store_rev"] = (transactions["revenue"] / transactions["store_total_revenue"]) * 100

# 3. Standardize revenue within each category (Category Z-Score)
cat_mean = transactions.groupby("category")["revenue"].transform("mean")
cat_std = transactions.groupby("category")["revenue"].transform("std")
transactions["category_zscore"] = (transactions["revenue"] - cat_mean) / cat_std

print("Transactions with Group-Transformed Metrics:")
print(transactions[["store_id", "category", "revenue", "store_total_revenue", "pct_of_store_rev", "category_zscore"]].round(2))


# ==============================================================================
# 4. GROUP FILTERING VIA .filter() (COHORT PRUNING)
# ==============================================================================
# `.filter()` retains or discards ENTIRE groups based on a boolean predicate
# evaluated on the group DataFrame.

print_separator("4. Cohort Pruning with GroupBy .filter()")

# Filter to keep only stores whose cumulative revenue exceeds $1,500
high_volume_stores = transactions.groupby("store_id").filter(lambda group: group["revenue"].sum() > 1500.0)

print(f"Original transaction count: {len(transactions)}")
print(f"Transactions from high-volume stores (> $1,500 total sales): {len(high_volume_stores)}")
print(high_volume_stores[["store_id", "category", "revenue"]])


# ==============================================================================
# 5. RELATIONAL JOINS: INNER, LEFT, RIGHT, FULL OUTER & CROSS JOINS
# ==============================================================================
# Pandas `pd.merge()` implements SQL-style relational database algebra.

print_separator("5. Relational Merges: Inner, Left, Outer & Cross Joins")

customers = pd.DataFrame({
    "cust_id": ["C1", "C2", "C3", "C4"],
    "name": ["Alice", "Bob", "Charlie", "David"],
    "tier": ["Gold", "Silver", "Platinum", "Bronze"]
})

orders = pd.DataFrame({
    "order_id": ["O101", "O102", "O103", "O104", "O105"],
    "cust_id": ["C1", "C2", "C1", "C5", "C2"],  # Note: C5 does not exist in customers!
    "amount": [250.0, 75.0, 110.0, 430.0, 89.0]
})

print("Customers Table:")
print(customers)
print("\nOrders Table (Notice C5 is unknown):")
print(orders)

# 1. Inner Join: intersection of keys (only matching cust_ids)
inner_merged = pd.merge(customers, orders, on="cust_id", how="inner")
print("\n1. Inner Join (Only customers with orders & known IDs):")
print(inner_merged[["cust_id", "name", "order_id", "amount"]])

# 2. Left Join: keep ALL customers, even those with zero orders (C3, C4)
left_merged = pd.merge(customers, orders, on="cust_id", how="left")
print("\n2. Left Join (All customers; missing orders become NaN):")
print(left_merged[["cust_id", "name", "order_id", "amount"]])

# 3. Outer Join: keep ALL customers AND all orphaned orders
outer_merged = pd.merge(customers, orders, on="cust_id", how="outer")
print("\n3. Full Outer Join (Union of all records):")
print(outer_merged[["cust_id", "name", "order_id", "amount"]])


# ==============================================================================
# 6. MERGE PROVENANCE AUDITING & INTEGRITY VALIDATION
# ==============================================================================
# `indicator=True`: adds a `_merge` column showing if row came from left, right, or both.
# `validate="1:m"`: validates relationship cardinality to prevent accidental Cartesian explosions!

print_separator("6. Merge Provenance Auditing & Schema Validation")

# Audit orphaned records
audit_merge = pd.merge(customers, orders, on="cust_id", how="outer", indicator=True, validate="1:m")
print("Provenance Breakdown (_merge count):")
print(audit_merge["_merge"].value_counts())

# Identify customers who NEVER placed an order
non_buyers = audit_merge.query("_merge == 'left_only'")[["cust_id", "name", "tier"]]
print("\nCustomers with Zero Orders (Churn/Acquisition target):")
print(non_buyers)

# Identify orphaned orders with invalid customer IDs
orphaned_orders = audit_merge.query("_merge == 'right_only'")[["order_id", "cust_id", "amount"]]
print("\nOrphaned Orders (Data Integrity Alert!):")
print(orphaned_orders)


# ==============================================================================
# 7. MULTIDIMENSIONAL CONCATENATION (pd.concat)
# ==============================================================================
# `axis=0`: Stacks rows vertically (union of logs/batches).
# `axis=1`: Merges columns horizontally by index alignment.

print_separator("7. Multidimensional Concatenation: Stacking Batches")

jan_sales = pd.DataFrame({"item": ["Pen", "Notebook"], "units": [100, 45]}, index=["J1", "J2"])
feb_sales = pd.DataFrame({"item": ["Pen", "Notebook", "Marker"], "units": [120, 60, 30]}, index=["F1", "F2", "F3"])

# Vertical concatenation with multi-index key tracking
quarter_df = pd.concat([jan_sales, feb_sales], keys=["January", "February"])
print("Vertically Concatenated Sales with Month Hierarchy:")
print(quarter_df)


# ==============================================================================
# 8. TIME-SERIES RESAMPLING, ROLLING WINDOWS & LAG FEATURES
# ==============================================================================
# Analyzing trends over time using rolling windows and period shifts.

print_separator("8. Time-Series Analysis: Rolling Windows & Lag Shifts")

# Generate 30 daily timestamps with synthetic sales data
rng = np.random.default_rng(99)
dates = pd.date_range(start="2024-01-01", periods=14, freq="D")
daily_sales = pd.DataFrame({
    "date": dates,
    "revenue": rng.integers(low=1000, high=5000, size=len(dates))
}).set_index("date")

# 1. Rolling 3-day simple moving average (SMA)
daily_sales["sma_3d"] = daily_sales["revenue"].rolling(window=3).mean()

# 2. Rolling standard deviation (Volatility)
daily_sales["volatility_3d"] = daily_sales["revenue"].rolling(window=3).std()

# 3. Lag Feature (Yesterday's revenue) via shift(1)
daily_sales["prev_day_revenue"] = daily_sales["revenue"].shift(1)

# 4. Day-over-Day (DoD) growth rate percentage
daily_sales["dod_growth_pct"] = daily_sales["revenue"].pct_change() * 100

print("Daily Time-Series Metrics (Rolling SMA & Day-over-Day Growth):")
print(daily_sales.round(2).head(10))


# ==============================================================================
# 9. PRODUCTION CASE STUDY: OMNICHANNEL RETAIL PERFORMANCE ENGINE
# ==============================================================================
print_separator("9. Real-World Case Study: Omnichannel Performance & Ranking")

stores_dim = pd.DataFrame({
    "store_id": ["S1", "S2", "S3"],
    "region": ["North America", "Europe", "Asia-Pacific"],
    "manager": ["Sarah Connor", "John Matrix", "Ellen Ripley"]
})

# Complete end-to-end analytical query
# 1. Aggregate store totals
store_stats = (
    transactions
    .groupby("store_id")
    .agg(
        total_rev=("revenue", "sum"),
        total_units=("units", "sum"),
        avg_price_per_unit=("revenue", lambda s: s.sum() / transactions.loc[s.index, "units"].sum())
    )
    .reset_index()
)

# 2. Join with Store Dimension Metadata
performance_report = (
    pd.merge(stores_dim, store_stats, on="store_id", how="inner")
    .assign(
        rev_share_pct=lambda d: (d["total_rev"] / d["total_rev"].sum()) * 100,
        sales_rank=lambda d: d["total_rev"].rank(ascending=False).astype(int)
    )
    .sort_values(by="sales_rank")
)

print("Executive Store Performance & Ranking Report:")
print(performance_report[["sales_rank", "store_id", "region", "manager", "total_rev", "rev_share_pct", "avg_price_per_unit"]].to_string(index=False))

print("================================================================================")
print(" Grouping, Aggregations & Relational Merges Masterclass Completed! ")
print("================================================================================")
