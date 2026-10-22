# Day 25: Python Data Analysis Masterclass 📊⚡

Welcome to **Day 25** of the Python Mastery Series! Today's masterclass focuses on **Data Analysis in Python**, covering foundational numerical computing with **NumPy**, tabular data manipulation with **Pandas**, rigorous data cleaning & statistical wrangling, relational joins, time-series analysis, and enterprise-grade analytical architectures.

```
                         THE MODERN PYTHON DATA ANALYSIS LIFECYCLE
                         
 ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
 │    Raw Ingestion     │      │   Data Cleaning &    │      │  Transformation &    │
 │  CSV / Parquet / SQL │ ───► │      Wrangling       │ ───► │ Feature Engineering  │
 │  JSON / REST APIs    │      │  Nulls, IQR, Outliers│      │  Groupby, Merges, DT │
 └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
                                                                        │
                                                                        ▼
 ┌──────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
 │ Actionable Decisions │      │ Executive Reporting  │      │ Exploratory Data     │
 │ Automated Triaging   │ ◄─── │ Tearsheets, Cohorts  │ ◄─── │    Analysis (EDA)    │
 │ Quantitative Models  │      │ Aggregation Tables   │      │ Summary Stats & Corr │
 └──────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

---

## 📁 Repository Structure

```text
DAY25-PYTHON/
│
├── 01_numpy_numerical_computing_and_vectorization.py  # Memory contiguity, strides, broadcasting rules, ufuncs, linear algebra & SIMD benchmark
├── 02_pandas_series_dataframes_and_indexing.py         # Series/DataFrame anatomy, .loc vs .iloc vs .query, SettingWithCopyWarning & memory optimization
├── 03_data_cleaning_wrangling_and_preprocessing.py     # Missing data mechanisms (MCAR/MAR/MNAR), IQR & Z-score outliers, regex cleaning & pivot tables
├── 04_grouping_aggregations_and_data_merging.py        # Split-Apply-Combine, named aggregations, .transform, relational joins, merge validation & rolling windows
├── 05_practical_projects.py                            # 3 Production Systems: Customer RFM Segmentation, Algorithmic Trading Backtester & ICU Telemetry Pipeline
├── .gitignore                                          # Standard Python bytecode, cache, venv, and IDE exclusions
└── README.md                                           # Architectural handbook, visual diagrams, tables & complete analytical cheat-sheet
```

---

## 🧭 Why Master Python for Data Analysis?

Data in the real world is messy, incomplete, multi-dimensional, and massive. Standard Python lists and procedural loops fail on large datasets due to interpreter boxing overhead, pointer indirection, and dynamic type checks. 

The scientific Python stack (**NumPy** and **Pandas**) bypasses the Global Interpreter Lock (GIL) and Python object overhead by delegating intensive computations directly to low-level C/Fortran routines, SIMD vector registers, and cache-aligned memory blocks.

### The Problem: Slow Procedural Python Loops

```python
# Procedural Python list computation (Anti-Pattern on tabular data):
def calculate_markup(prices: list[float], tax: float) -> list[float]:
    result = []
    for p in prices:
        result.append(p * 1.20 + tax)
    return result  # High CPU cache miss rate, 20x to 100x slower
```

### The Solution: SIMD-Accelerated Vectorization

```python
import numpy as np

# Vectorized C execution across contiguous memory:
def calculate_markup_vectorized(prices: np.ndarray, tax: float) -> np.ndarray:
    return prices * 1.20 + tax  # Single instruction multiple data (SIMD)
```

---

## 🏛️ Memory Architecture: Python Lists vs. NumPy ndarrays

Understanding memory layout is the key to writing high-performance data analysis pipelines.

```
 Python List Memory Layout (Array of Pointers):
 ┌───────────────┐
 │ List Object   │ ──► [Ptr 0] ──────► PyObject (Integer 10, ~28 bytes)
 │ (Header, len) │ ──► [Ptr 1] ──────► PyObject (Integer 20, ~28 bytes)
 └───────────────┘ ──► [Ptr 2] ──────► PyObject (Integer 30, ~28 bytes)
                       (Scattered across arbitrary heap memory addresses)

 NumPy ndarray Memory Layout (Contiguous Memory Buffer):
 ┌───────────────┐
 │ ndarray Header│ ──► [ 10 | 20 | 30 ]  (Sequential 8-byte int64 block)
 │ Shape, Strides│     (Directly cacheable in CPU L1/L2/L3 registers!)
 └───────────────┘
```

---

## 📚 Core Concepts & Deep-Dive Breakdown

### 1. NumPy Broadcasting Rules

Broadcasting enables arithmetic operations between arrays of differing dimensions without memory duplication.

**The Formal Broadcasting Rule:**
When operating on two arrays, NumPy compares their shapes element-wise starting from the **trailing (rightmost) dimensions** and working backwards:
Two dimensions are compatible if:
1. They are equal, **OR**
2. One of them is 1.

```
Array A (2D):   3 x 3
Array B (1D):       3   ──► Broadcasted to 1 x 3 ──► Stretched to 3 x 3
-----------------------------------------------------------------------
Result Shape:   3 x 3   (Evaluated element-wise without memory copying)
```

---

### 2. The Dreaded `SettingWithCopyWarning` in Pandas

Beginners frequently encounter:
`SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.`

#### Why does this happen?
When writing `df[df['dept'] == 'Sales']['bonus'] = 5000`, the operation is chained into two separate steps:
1. `__getitem__`: `df[df['dept'] == 'Sales']` (Might return a VIEW or a COPY depending on memory alignment).
2. `__setitem__`: `['bonus'] = 5000`

If step 1 returns a copy, you are modifying a temporary object that is immediately garbage collected, leaving `df` unchanged!

#### The Two Golden Solutions:

```python
# ANTI-PATTERN (Triggers SettingWithCopyWarning):
sales = df[df['department'] == 'Sales']
sales['bonus'] = 5000

# SOLUTION 1: Explicitly isolate the sub-DataFrame using .copy()
sales = df[df['department'] == 'Sales'].copy()
sales['bonus'] = 5000  # Safe!

# SOLUTION 2: Modify the original DataFrame in a single step with .loc
df.loc[df['department'] == 'Sales', 'bonus'] = 5000  # Safe and explicit!
```

---

### 3. The Split-Apply-Combine Paradigm

Hadley Wickham's foundational concept powers `df.groupby()`:

```
                      SPLIT                 APPLY                COMBINE
                                        (Sum Revenue)
                 ┌── Store S1 [1200, 350] ──► $1,550 ──┐
                 │                                     │
 Raw Data Table ─┼── Store S2 [950, 210]  ──► $1,160 ──┼──► Aggregate Summary
 (All Stores)    │                                     │    Table by Store
                 └── Store S3 [1850, 600] ──► $2,450 ──┘
```

* **Aggregation (`.agg`)**: Collapses multiple rows into a single scalar value per group.
* **Transformation (`.transform`)**: Retains original row shape; useful for group percentages, running totals, and group z-scores.
* **Filtration (`.filter`)**: Retains or discards entire groups based on a collective boolean condition.

---

### 4. Relational Merges Venn Diagram (`pd.merge`)

```
       INNER JOIN                   LEFT JOIN                   FULL OUTER JOIN
    ┌───────┬───────┐           ┌───────┬───────┐           ┌───────┬───────┐
    │       │XXXXXXX│           │XXXXXXX│XXXXXXX│           │XXXXXXX│XXXXXXX│XXXXXXX│
    │ Left  │ Match │           │ Left  │ Match │           │ Left  │ Match │ Right │
    │       │XXXXXXX│           │XXXXXXX│XXXXXXX│           │XXXXXXX│XXXXXXX│XXXXXXX│
    └───────┴───────┘           └───────┴───────┘           └───────┴───────┘
  Only matching keys          All Left + Matched Right      All Left + All Right
```

Always use `indicator=True` to audit merge provenance and diagnose orphaned records!

---

## 🛠️ Complete Analytical Cheat Sheet

### NumPy Core Operations

| Method | Description | Example |
| :--- | :--- | :--- |
| `np.arange(start, stop, step)` | Generates range array | `np.arange(0, 10, 2)` |
| `np.linspace(start, stop, n)` | Generates $n$ evenly spaced points | `np.linspace(0, 1, 5)` |
| `np.random.default_rng(seed)` | Modern PCG64 random generator | `rng.normal(0, 1, size=(3, 3))` |
| `arr.reshape(rows, cols)` | Reshapes without data copy | `arr.reshape(2, -1)` |
| `np.where(cond, x, y)` | Vectorized ternary conditional | `np.where(arr > 0, arr, 0)` |
| `np.select(condlist, choicelist)` | Multi-branch categorical classifier | `np.select([a < 10, a >= 10], ['Low', 'High'])` |
| `arr.mean(axis=0/1)` | Reduction along column/row axis | `arr.mean(axis=0, keepdims=True)` |
| `A @ B` or `np.matmul(A, B)` | Matrix multiplication operator | `weights @ cov_matrix @ weights` |

---

### Pandas Indexing & Transformation

| Syntax | Type | Behavior | Example |
| :--- | :--- | :--- | :--- |
| `df.loc[rows, cols]` | Label-Based | Inclusive of both start and end labels | `df.loc['R1':'R5', ['A', 'B']]` |
| `df.iloc[rows, cols]` | Integer Position | 0-indexed, exclusive of end position | `df.iloc[0:5, 0:2]` |
| `df.query("expression")` | String Expression | SQL-like filtering with `@var` reference | `df.query("age > 30 and dept == 'Sales'")` |
| `df.drop_duplicates()` | Deduplication | Drops duplicate rows based on subset | `df.drop_duplicates(subset=['email'])` |
| `df.dropna(how, thresh)` | Null Removal | Drops rows/cols based on missingness | `df.dropna(thresh=5)` |
| `df.pivot_table()` | Reshaping | Aggregates long data into wide matrix | `df.pivot_table(index='store', columns='qtr')` |
| `pd.melt()` | Reshaping | Unpivots wide columns into long rows | `pd.melt(df, id_vars=['id'])` |
| `df.groupby().agg()` | Aggregation | Named aggregation tuple syntax | `.agg(avg_rev=('revenue', 'mean'))` |

---

## 🚀 Execution Instructions

Run every module directly from the repository root:

```bash
# 1. NumPy Numerical Computing & Vectorization
python 01_numpy_numerical_computing_and_vectorization.py

# 2. Pandas Series, DataFrames & Indexing Mastery
python 02_pandas_series_dataframes_and_indexing.py

# 3. Data Cleaning, Wrangling & Preprocessing
python 03_data_cleaning_wrangling_and_preprocessing.py

# 4. Grouping, Aggregations & Relational Merges
python 04_grouping_aggregations_and_data_merging.py

# 5. Production Systems (RFM Segmentation, Trading Backtester, ICU Telemetry)
python 05_practical_projects.py
```

---

## 🏆 Production Projects Implemented in `05_practical_projects.py`

### Project 1: E-Commerce RFM Segmentation & Cohort Profiler
* Ingests raw transaction streams, calculates customer Recency, Frequency, and Monetary scores.
* Applies quantile-based scoring (`qcut`) into formal cohorts: **Champions**, **Loyal Customers**, **At Risk**, and **Lost**.
* Generates an executive BI report with revenue share attribution per behavioral segment.

### Project 2: Quantitative Finance & Algorithmic Trading Analyzer
* Simulates 1 year of daily multi-asset price trajectories using **Geometric Brownian Motion (GBM)**.
* Implements a vectorized dual **Exponential Moving Average (EMA 12/26)** crossover trend-following strategy.
* Generates an institutional risk tearsheet computing **CAGR**, **Annualized Volatility**, **Sharpe Ratio (Rf=4%)**, **Maximum Drawdown (MDD)**, and **Win Hit Rate**.

### Project 3: Healthcare Clinical Patient Telemetry & Anomaly Detector
* Ingests high-frequency ICU sensor feeds (Heart Rate, Blood Oxygen SpO2, Systolic BP, Temperature).
* Cleans telemetry dropouts using time-bounded forward-fill (`ffill(limit=3)`).
* Evaluates a dual-layer anomaly engine: hard physiological thresholds and statistical **Median Absolute Deviation (MAD)**.
* Automatically dispatches real-time **CRITICAL** triage alerts for septic shock, hypoxemia, and acute arrhythmias.

---

## 💡 Best Practices for Production Data Analysis

1. **Avoid In-Place Mutation (`inplace=True`)**: In modern Pandas (2.0+), `inplace=True` is actively discouraged and often does not provide performance gains. Prefer functional method chaining (`.assign()`, `.pipe()`).
2. **Optimize Memory Early**: Downcast default `int64` and `float64` types and convert repeated string dimensions to `category`. This routinely yields an **80–95% RAM reduction**.
3. **Never Iterate Rows with Loops**: Never use Python `for` loops or `.iterrows()` for calculations. Always use vectorized Series operations or `.apply()` as a last resort.
4. **Guard Against Lookahead Bias**: In financial and time-series modeling, always shift signals by at least 1 period (`.shift(1)`) so decisions are executed only on known past information.
5. **Always Audit Merges**: Use `indicator=True` and `validate="1:m"` in `pd.merge()` to ensure your joins do not silently produce duplicate row multiplications.

---

*Part of the Python Mastery Series.*
