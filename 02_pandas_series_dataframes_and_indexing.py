"""
================================================================================
Day 25: Python Data Analysis - Pandas Series, DataFrames & Indexing Mastery
================================================================================
Pandas is Python's premier library for structured, labeled tabular data analysis.
Built on top of NumPy, it adds metadata (row labels, column names, flexible types,
and temporal indices) while maintaining vectorized C performance.

Theoretical Background:
A Pandas `DataFrame` is essentially a collection of aligned 1-dimensional `Series`
objects sharing a common `Index`. Under the hood:
  - Each `Series` is backed by a NumPy 1D array or an ExtensionArray.
  - Data is organized into heterogeneous column blocks (the BlockManager/ArrayManager).
  - Row labels (`index`) provide O(1) hash-based lookups and automated alignment
    during arithmetic operations.

Key Topics Covered in this Script:
1. Core Mental Model: Series, DataFrames, and the Anatomy of the Index
2. Data Ingestion & Creation from Structured Dicts, CSV Streams & JSON
3. Indexing Mastery: `.loc` (Label) vs. `.iloc` (Position) vs. `.query()`
4. Demystifying `SettingWithCopyWarning`: Chained Assignment Traps & Solutions
5. Schema Profiling & Structural Inspection (`info`, `describe`, `value_counts`)
6. Memory Footprint Optimization: Downcasting & Categorical Dtypes
7. High-Performance Vectorized String (`.str`) and Datetime (`.dt`) Accessors
8. Transformations via `.map()`, `.apply()`, and Method Chaining with `.assign()`
9. Real-World Case Study: Customer Churn & Lifetime Value (LTV) Profiler
================================================================================
"""

import io
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any


def print_separator(title: str) -> None:
    """Helper to print formatted section banners for clean console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. CORE ARCHITECTURE: SERIES, DATAFRAMES & AUTOMATIC INDEX ALIGNMENT
# ==============================================================================
# A Series is a 1D labeled array. When performing operations between two Series,
# Pandas aligns them based on index labels, NOT positional order!

print_separator("1. Architecture: Series & Automated Index Alignment")

q1_revenue = pd.Series([120_000, 150_000, 95_000], index=["North", "South", "East"], name="Q1_Revenue")
q2_revenue = pd.Series([110_000, 130_000, 160_000], index=["East", "North", "West"], name="Q2_Revenue")

print("Q1 Revenue (North, South, East):")
print(q1_revenue)
print("\nQ2 Revenue (East, North, West - Different Order & Regions):")
print(q2_revenue)

# Automatic alignment by label:
total_h1 = q1_revenue + q2_revenue
print("\nTotal H1 Revenue (Notice North & East added properly, South/West become NaN):")
print(total_h1)

# Resolving missing alignments gracefully with .add(fill_value=0)
total_h1_filled = q1_revenue.add(q2_revenue, fill_value=0)
print("\nTotal H1 Revenue with fill_value=0 (No NaNs):")
print(total_h1_filled)


# ==============================================================================
# 2. DATA INGESTION & DATAFRAME CREATION
# ==============================================================================
# Data can be ingested from Python dicts, records, or streaming CSV strings.

print_separator("2. DataFrame Creation & CSV Ingestion")

csv_data = """employee_id,name,department,salary,hire_date,performance_score
E101, Alice Smith,Engineering,115000,2021-03-15,4.8
E102, Bob Johnson,Marketing,78000,2022-06-01,3.9
E103, Charlie Lee,Engineering,128000,2020-01-10,4.9
E104, Diana Evans,Sales,92000,2023-08-20,4.2
E105, Evan Wright,Marketing,81000,2021-11-05,3.7
E106, Fiona Gallagher,Engineering,105000,2022-02-18,4.5
"""

# Parse directly from an in-memory CSV stream
df = pd.read_csv(
    io.StringIO(csv_data),
    parse_dates=["hire_date"],
    index_col="employee_id"
)

# Strip any whitespace from string columns
df["name"] = df["name"].str.strip()
df["department"] = df["department"].str.strip()

print("Parsed Employee DataFrame:")
print(df)


# ==============================================================================
# 3. INDEXING MASTERY: .loc VS .iloc VS .query()
# ==============================================================================
# Rule of Thumb:
#   - `.loc[rows, cols]`: Access by LABEL (inclusive start AND inclusive stop!).
#   - `.iloc[rows, cols]`: Access by INTEGER POSITION (0-indexed, exclusive stop).
#   - `.query("expr")`: Evaluate boolean expression string with SQL-like readability.

print_separator("3. Indexing Mastery: .loc, .iloc, and .query()")

# 1. Label-based indexing with .loc
print("1. .loc['E101':'E103', ['name', 'salary']] (Notice E103 is INCLUDED):")
print(df.loc["E101":"E103", ["name", "salary"]])

# 2. Integer position indexing with .iloc
print("\n2. .iloc[0:2, 0:2] (Notice index 2 is EXCLUDED):")
print(df.iloc[0:2, 0:2])

# 3. Boolean Filtering with .loc
eng_high_earners = df.loc[(df["department"] == "Engineering") & (df["salary"] > 110_000), ["name", "salary"]]
print("\n3. Boolean Filtered (Engineering & Salary > 110k):")
print(eng_high_earners)

# 4. Expressive SQL-like .query()
# Use @ to reference external Python variables inside query strings!
min_score = 4.0
top_performers = df.query("performance_score >= @min_score and department != 'Marketing'")
print(f"\n4. .query() with external variable @min_score={min_score}:")
print(top_performers[["name", "department", "performance_score"]])


# ==============================================================================
# 4. DEMYSTIFYING SettingWithCopyWarning & CHAINED ASSIGNMENT
# ==============================================================================
# Chained indexing `df['col'][row_mask] = val` is an ANTI-PATTERN!
# Why? Because `df['col']` might return a COPY or a VIEW depending on memory layout.
# If it returns a copy, modifying it mutates a temporary object that is immediately
# garbage-collected, leaving your original DataFrame untouched!

print_separator("4. Demystifying SettingWithCopyWarning")

# Create a sample slice
sales_team = df[df["department"] == "Sales"]

print("Attempting chained assignment: sales_team['bonus'] = 5000")
# The wrong way:
with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always")
    # This triggers SettingWithCopyWarning
    sales_team["bonus"] = 5000
    if caught_warnings:
        print(f"  [CAUGHT WARNING]: {caught_warnings[0].category.__name__}")
        print(f"  Message: {caught_warnings[0].message}")

# The correct way 1: Explicit copy when creating a standalone subset
sales_team_safe = df[df["department"] == "Sales"].copy()
sales_team_safe["bonus"] = 5000
print(f"\nCorrect Way 1 (.copy()): Bonus added successfully without warning!")
print(sales_team_safe[["name", "salary", "bonus"]])

# The correct way 2: Modify the original DataFrame in-place using .loc
df.loc[df["department"] == "Sales", "salary"] += 3000
print(f"\nCorrect Way 2 (.loc on original df): Updated Sales salary in original df:")
print(df.loc["E104", ["name", "department", "salary"]])


# ==============================================================================
# 5. SCHEMA PROFILING & STRUCTURAL INSPECTION
# ==============================================================================
# Key diagnostic utilities: `.info()`, `.describe()`, and `.value_counts()`

print_separator("5. Schema Profiling & Structural Diagnostics")

print("1. DataFrame Info Summary:")
df.info()

print("\n2. Numerical Summary Statistics (.describe()):")
print(df.describe().T)

print("\n3. Categorical Value Counts (Department Distribution %):")
dept_dist = df["department"].value_counts(normalize=True) * 100
for dept, pct in dept_dist.items():
    print(f"  {dept:>12}: {pct:5.1f}%")


# ==============================================================================
# 6. MEMORY FOOTPRINT OPTIMIZATION: DTYPE DOWNCASTING & CATEGORICALS
# ==============================================================================
# Default pandas dtypes (int64, float64, object strings) consume excessive memory.
# Converting repeated strings to `category` and downcasting integers can save 80%+ RAM.

print_separator("6. Memory Optimization: Downcasting & Categoricals")

# Create a larger synthetic dataset to clearly demonstrate memory impact
n_records = 50_000
opt_df = pd.DataFrame({
    "user_id": np.random.randint(1, 1000, size=n_records),
    "tier": np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], size=n_records),
    "region": np.random.choice(["EMEA", "APAC", "AMER", "LATAM"], size=n_records),
    "score": np.random.uniform(0.0, 100.0, size=n_records)
})

initial_mem = opt_df.memory_usage(deep=True).sum() / (1024 * 1024)
print(f"Initial Dataset Memory (50k rows): {initial_mem:.2f} MB")

# Optimize:
# 1. user_id: int64 (8 bytes) -> int16 (2 bytes, supports up to 32,767)
opt_df["user_id"] = pd.to_numeric(opt_df["user_id"], downcast="integer")

# 2. tier & region: object (strings) -> category (internal integer lookup table)
opt_df["tier"] = opt_df["tier"].astype("category")
opt_df["region"] = opt_df["region"].astype("category")

# 3. score: float64 (8 bytes) -> float32 (4 bytes)
opt_df["score"] = pd.to_numeric(opt_df["score"], downcast="float")

optimized_mem = opt_df.memory_usage(deep=True).sum() / (1024 * 1024)
reduction_pct = ((initial_mem - optimized_mem) / initial_mem) * 100

print(f"Optimized Dataset Memory        : {optimized_mem:.2f} MB")
print(f"Memory Footprint Reduction      : {reduction_pct:.1f}% SAVED!")


# ==============================================================================
# 7. VECTORIZED STRING (.str) & DATETIME (.dt) ACCESSORS
# ==============================================================================
# Avoid slow Python for-loops over strings or dates; use vectorized accessors.

print_separator("7. Vectorized String (.str) & Datetime (.dt) Accessors")

# String operations
df["email"] = (df["name"].str.lower().str.replace(" ", ".", regex=False) + "@company.com")
df["initials"] = df["name"].str.split(" ").str[0].str[0] + df["name"].str.split(" ").str[1].str[0]

print("Generated Vectorized Emails & Initials:")
print(df[["name", "initials", "email"]])

# Datetime accessors
df["hire_year"] = df["hire_date"].dt.year
df["hire_quarter"] = df["hire_date"].dt.to_period("Q")
df["tenure_days"] = (pd.Timestamp.now() - df["hire_date"]).dt.days

print("\nExtracted Temporal Components:")
print(df[["name", "hire_date", "hire_year", "hire_quarter", "tenure_days"]])


# ==============================================================================
# 8. FUNCTIONAL TRANSFORMATIONS: .map(), .apply() & .assign()
# ==============================================================================
# `.map()`: Maps values of a Series using a dictionary or mapping function.
# `.apply()`: Applies a function along an axis of a DataFrame or Series.
# `.assign()`: Method-chaining helper that returns a new DataFrame with new columns.

print_separator("8. Functional Transformations & Method Chaining")

# 1. .map() with a dictionary lookup
rating_labels = {3.7: "Satisfactory", 3.9: "Good", 4.2: "Very Good", 4.5: "Superior", 4.8: "Exemplary", 4.9: "Exceptional"}
df["rating_tier"] = df["performance_score"].map(rating_labels)

# 2. .apply() across rows (axis=1) for complex multi-column logic
def calculate_annual_bonus(row: pd.Series) -> float:
    base_salary = row["salary"]
    score = row["performance_score"]
    if score >= 4.5:
        return base_salary * 0.15
    elif score >= 4.0:
        return base_salary * 0.10
    else:
        return base_salary * 0.05

df["calculated_bonus"] = df.apply(calculate_annual_bonus, axis=1)

print("Mapped Rating Tiers & Calculated Bonuses:")
print(df[["name", "salary", "performance_score", "rating_tier", "calculated_bonus"]])


# ==============================================================================
# 9. REAL-WORLD CASE STUDY: CUSTOMER LIFETIME VALUE & CHURN SCORER
# ==============================================================================
print_separator("9. Real-World Case Study: Customer Lifetime Value (LTV)")

customers_data = {
    "customer_id": [f"C{1000+i}" for i in range(8)],
    "signup_date": pd.to_datetime(["2022-01-15", "2022-03-22", "2021-11-05", "2023-01-10",
                                  "2020-07-19", "2023-05-30", "2021-08-14", "2022-12-01"]),
    "total_orders": [14, 3, 28, 5, 45, 2, 19, 8],
    "total_spend": [1450.50, 180.00, 3200.75, 450.20, 5800.00, 120.50, 2100.00, 890.30],
    "last_order_days_ago": [12, 145, 8, 95, 4, 210, 35, 60]
}

cust_df = pd.DataFrame(customers_data).set_index("customer_id")

# Vectorized analytics pipeline using method chaining (.assign())
current_date = pd.Timestamp("2024-01-01")

analyzed_cust = (
    cust_df
    .assign(
        aov=lambda d: d["total_spend"] / d["total_orders"],  # Average Order Value
        tenure_months=lambda d: ((current_date - d["signup_date"]).dt.days / 30.4).round(1),
        is_churned=lambda d: d["last_order_days_ago"] > 90,
        churn_risk=lambda d: np.where(
            d["last_order_days_ago"] > 120, "High Risk",
            np.where(d["last_order_days_ago"] > 45, "Medium Risk", "Active")
        ),
        ltv_segment=lambda d: pd.qcut(
            d["total_spend"], q=3, labels=["Low Value", "Medium Value", "High Value"]
        )
    )
)

print("Comprehensive Customer Analytics Report:")
print(analyzed_cust[["total_orders", "total_spend", "aov", "last_order_days_ago", "churn_risk", "ltv_segment"]])
print("\nSegment Breakdown by Spend Tier:")
print(analyzed_cust.groupby("ltv_segment", observed=False)[["total_spend", "aov"]].mean().round(2))

print("================================================================================")
print(" Pandas Series, DataFrames & Indexing Masterclass Completed! ")
print("================================================================================")
