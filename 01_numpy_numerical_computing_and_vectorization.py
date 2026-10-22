"""
================================================================================
Day 25: Python Data Analysis - NumPy Numerical Computing & Vectorization
================================================================================
NumPy (Numerical Python) is the foundational library for scientific computing
and data analysis in Python. It provides the high-performance N-dimensional array
object (ndarray) and tools for working with arrays in contiguous memory buffers.

Theoretical Background:
Standard Python lists are arrays of pointers pointing to disparate heap-allocated
Python objects (each with reference count, type info, and value overhead).
NumPy `ndarray` stores elements in a single contiguous block of C-allocated memory
with a homogeneous data type (dtype). This architecture unlocks:
  1. Cache Locality (L1/L2/L3 CPU cache efficiency during sequential reads).
  2. SIMD Vectorization (Single Instruction, Multiple Data CPU registers).
  3. Elimination of dynamic Python type-checking overhead inside compute loops.

Key Topics Covered in this Script:
1. Architecture & Memory Contiguity: Python Lists vs. C-Contiguous ndarrays
2. Array Creation Mechanics & Random Generator Seed Reproducibility
3. Indexing, Slicing, and the Crucial Difference: Views vs. Deep Copies
4. Advanced Boolean Masking, Fancy Indexing, `np.where()` & `np.select()`
5. The Formal Rules of Array Broadcasting (Dimension Expansion & Alignment)
6. Universal Functions (ufuncs), Reductions, and the `axis` Parameter
7. Matrix Operations & Linear Algebra (`@`, `np.matmul()`, `np.linalg`)
8. Performance Benchmark: Pure Python Loops vs. Vectorized NumPy Execution
9. Real-World Quantitative Case Study: Vectorized Multi-Asset Portfolio Risk
================================================================================
"""

import sys
import time
import numpy as np
from typing import Tuple


def print_separator(title: str) -> None:
    """Helper to print formatted section banners for clean console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. ARCHITECTURE & MEMORY CONTIGUITY: LISTS VS NDARRAY
# ==============================================================================
# In a Python list `[1, 2, 3]`:
# - The list itself is an array of 8-byte 64-bit pointers.
# - Each integer is a PyObject taking ~28 bytes on 64-bit systems.
# - Elements are scattered arbitrarily in heap memory -> Cache misses!
#
# In a NumPy array `np.array([1, 2, 3], dtype=np.int64)`:
# - All values reside sequentially in continuous bytes (8 bytes each).
# - Strides tell the engine how many bytes to jump to reach the next element.

print_separator("1. Architecture & Memory: Python List vs. NumPy ndarray")

py_list = list(range(1000))
np_arr = np.arange(1000, dtype=np.int64)

# Memory consumption comparison
list_pointers_size = sys.getsizeof(py_list)
individual_element_size = sys.getsizeof(py_list[0])
total_list_est = list_pointers_size + (individual_element_size * len(py_list))

print(f"Python List (1,000 ints) estimated heap size : ~{total_list_est:,} bytes")
print(f"NumPy Array (1,000 int64) continuous buffer size: {np_arr.nbytes:,} bytes")
print(f"Array Metadata:")
print(f"  - Shape    : {np_arr.shape}")
print(f"  - Dtype    : {np_arr.dtype}")
print(f"  - Itemsize : {np_arr.itemsize} bytes per element")
print(f"  - Strides  : {np_arr.strides} (bytes to step along each axis)")
print(f"  - C-Flags  : Contiguous = {np_arr.flags.c_contiguous}")


# ==============================================================================
# 2. ARRAY CREATION & REPRODUCIBLE PSEUDO-RANDOM NUMBER GENERATION
# ==============================================================================
# NumPy 1.17+ introduced `np.random.default_rng(seed)` which implements PCG64,
# replacing the legacy `np.random.seed()` / Mersenne Twister for cryptographic
# and statistical robustness.

print_separator("2. Array Creation Mechanics & Modern Random Generators")

# Explicit builders
zeros_2d = np.zeros(shape=(2, 3), dtype=np.float64)
ones_3d = np.ones(shape=(2, 2, 2), dtype=np.int32)
identity_matrix = np.eye(3, dtype=np.float32)
linear_spaced = np.linspace(start=0.0, stop=1.0, num=5)  # 5 evenly spaced points
stepped_range = np.arange(start=10, stop=30, step=5)     # Exclusive stop

print(f"2x3 Zeros:\n{zeros_2d}")
print(f"3x3 Identity Matrix:\n{identity_matrix}")
print(f"Linspace (0.0 to 1.0, 5 points): {linear_spaced}")
print(f"Arange (10 to 30, step 5)       : {stepped_range}")

# Modern Random Generator (PCG64)
rng = np.random.default_rng(seed=42)
normal_dist = rng.normal(loc=100.0, scale=15.0, size=(2, 4))  # Mean=100, Std=15
integers = rng.integers(low=1, high=100, size=5)

print(f"\nNormally Distributed Data (mean=100, std=15):\n{np.round(normal_dist, 2)}")
print(f"Random Integers [1, 100): {integers}")


# ==============================================================================
# 3. INDEXING, SLICING, VIEWS VS. DEEP COPIES
# ==============================================================================
# CRITICAL PITFALL: Slicing a NumPy array creates a VIEW (shallow memory window),
# NOT a copy! Modifying a slice directly mutates the original parent array!
# To isolate changes, always use `.copy()`.

print_separator("3. Indexing, Slicing, Views vs. Copies Trap")

matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

print(f"Original Matrix:\n{matrix}")

# Slice sub-matrix: rows 0 to 2, columns 1 to 3
slice_view = matrix[0:2, 1:3]
print(f"\nSub-matrix slice (Rows 0-1, Cols 1-2):\n{slice_view}")

# Mutating the view mutates original!
slice_view[0, 0] = 9999
print(f"After modifying slice_view[0, 0] = 9999:")
print(f"Does slice_view share memory with matrix? {slice_view.base is matrix}")
print(f"Original Matrix modified value at [0, 1]: {matrix[0, 1]} (MUTATED!)")

# Reset and demonstrate safe .copy()
matrix[0, 1] = 20
safe_copy = matrix[0:2, 1:3].copy()
safe_copy[0, 0] = -111
print(f"After safe_copy[0, 0] = -111:")
print(f"Does safe_copy share memory? {safe_copy.base is matrix}")
print(f"Original Matrix preserved value at [0, 1]: {matrix[0, 1]} (SAFE!)")


# ==============================================================================
# 4. ADVANCED BOOLEAN MASKING, FANCY INDEXING & CONDITIONAL REPLACEMENTS
# ==============================================================================
# Boolean masking evaluates conditions element-wise, returning a boolean array
# that can be used directly as an index filter.

print_separator("4. Boolean Masking, Fancy Indexing & np.where / np.select")

sensor_readings = np.array([23.5, -999.0, 24.1, 28.9, -999.0, 31.2, 19.8])
print(f"Raw Sensor Readings: {sensor_readings}")

# Multi-condition boolean masking (Note: MUST use bitwise & | ~, NOT and/or)
valid_mask = (sensor_readings != -999.0) & (sensor_readings >= 20.0)
print(f"Boolean Mask ((!= -999) & (>= 20)): {valid_mask}")
print(f"Filtered Valid Readings           : {sensor_readings[valid_mask]}")

# Fancy Indexing (indexing by integer arrays/lists)
indices = [0, 3, 5]
print(f"Fancy Indexed at {indices}           : {sensor_readings[indices]}")

# np.where(condition, if_true, if_false)
clean_readings = np.where(sensor_readings == -999.0, np.nan, sensor_readings)
print(f"Imputed with np.where (NaN replacement): {clean_readings}")

# np.select(conditions_list, choices_list, default) for multi-tier bucketing
temperatures = np.array([12.0, 22.5, 36.8, 41.2, 5.0])
conditions = [
    temperatures < 15.0,
    (temperatures >= 15.0) & (temperatures < 30.0),
    (temperatures >= 30.0) & (temperatures < 40.0)
]
labels = ["Cold", "Moderate", "Hot"]
classified = np.select(conditions, labels, default="Extreme Heat")
print(f"\nTemperatures: {temperatures}")
print(f"Classified Categories: {classified}")


# ==============================================================================
# 5. THE FORMAL RULES OF ARRAY BROADCASTING
# ==============================================================================
# Broadcasting allows arithmetic operations between arrays of different shapes
# without copying data, by matching dimensions from right to left (trailing).
#
# Formal Rule:
# Two dimensions are compatible when:
#   1. They are equal, OR
#   2. One of them is 1.
# If dimensions do not match and neither is 1, a ValueError is raised.

print_separator("5. Array Broadcasting Mechanics")

# Shape (3, 3) + Shape (3,) -> (3,) is treated as (1, 3) then stretched to (3, 3)
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
B = np.array([10, 20, 30])  # Shape (3,)

print(f"Matrix A (Shape {A.shape}):\n{A}")
print(f"Vector B (Shape {B.shape}): {B}")
print(f"A + B (Broadcasting B across each row):\n{A + B}")

# Column broadcasting: Shape (3, 1) + Shape (1, 4) -> Result Shape (3, 4)
col_vec = np.array([[100], [200], [300]])  # Shape (3, 1)
row_vec = np.array([[1, 2, 3, 4]])          # Shape (1, 4)
outer_sum = col_vec + row_vec
print(f"\nCol (3, 1) + Row (1, 4) -> Broadcasted Matrix (3, 4):\n{outer_sum}")


# ==============================================================================
# 6. UNIVERSAL FUNCTIONS (UFUNCS) & AXIS-WISE REDUCTIONS
# ==============================================================================
# `axis=0` operates down the columns (collapses rows).
# `axis=1` operates across the rows (collapses columns).
# `keepdims=True` preserves the rank for subsequent broadcast math.

print_separator("6. Universal Functions (ufuncs) & Reductions across Axes")

scores = np.array([
    [85, 92, 78],
    [79, 95, 88],
    [90, 82, 91],
    [68, 74, 80]
])  # 4 Students (rows), 3 Subjects (cols)

print(f"Scores Matrix (4 Students x 3 Subjects):\n{scores}")
print(f"Overall Mean: {scores.mean():.2f}")

# Axis 0 (column-wise / per subject):
subject_means = scores.mean(axis=0)
print(f"Subject Means (axis=0, across rows) : {np.round(subject_means, 2)}")

# Axis 1 (row-wise / per student):
student_totals = scores.sum(axis=1)
print(f"Student Totals (axis=1, across cols): {student_totals}")

# Standardization with keepdims=True (Z-Score normalization per column)
col_means = scores.mean(axis=0, keepdims=True)  # Shape (1, 3)
col_stds = scores.std(axis=0, keepdims=True)    # Shape (1, 3)
z_scores = (scores - col_means) / col_stds
print(f"\nZ-Score Normalized Scores (per column):\n{np.round(z_scores, 2)}")
print(f"Normalized column means (should be ~0): {np.round(z_scores.mean(axis=0), 4)}")
print(f"Normalized column stds  (should be ~1): {np.round(z_scores.std(axis=0), 4)}")


# ==============================================================================
# 7. LINEAR ALGEBRA & MATRIX OPERATIONS
# ==============================================================================
# In Python 3.5+, `@` is the dedicated matrix multiplication operator.
# NumPy's `np.linalg` provides optimized BLAS/LAPACK accelerated solvers.

print_separator("7. Matrix Multiplication & Linear Algebra (np.linalg)")

# Solve a system of linear equations:
#   2x + 3y = 8
#   3x + 4y = 11
# In matrix form: A @ X = B  =>  X = A^(-1) @ B
coeff_matrix = np.array([[2.0, 3.0], [3.0, 4.0]])
constants = np.array([8.0, 11.0])

solution = np.linalg.solve(coeff_matrix, constants)
print(f"Coefficient Matrix A:\n{coeff_matrix}")
print(f"Constants Vector B  : {constants}")
print(f"Exact Solution [x, y]: {solution}")

# Matrix properties
det = np.linalg.det(coeff_matrix)
inv = np.linalg.inv(coeff_matrix)
print(f"Determinant of A    : {det:.4f}")
print(f"Inverse of A        :\n{inv}")
print(f"Verification A @ inv(A) (Identity Matrix):\n{np.round(coeff_matrix @ inv, 2)}")


# ==============================================================================
# 8. PERFORMANCE BENCHMARK: PYTHON LOOPS VS. NUMPY VECTORIZATION
# ==============================================================================
# Compare running arithmetic operations on 1,000,000 elements.

print_separator("8. Performance Benchmark: Pure Python vs. NumPy SIMD")

N = 1_000_000
python_list_a = list(range(N))
python_list_b = list(range(N))
np_array_a = np.arange(N, dtype=np.float64)
np_array_b = np.arange(N, dtype=np.float64)

# 1. Benchmark Pure Python Loop
start_time = time.perf_counter()
py_result = [a * 2.5 + b for a, b in zip(python_list_a, python_list_b)]
py_elapsed = time.perf_counter() - start_time

# 2. Benchmark Vectorized NumPy
start_time = time.perf_counter()
np_result = np_array_a * 2.5 + np_array_b
np_elapsed = time.perf_counter() - start_time

speedup = py_elapsed / np_elapsed if np_elapsed > 0 else float("inf")

print(f"Processing {N:,} elements (c = a * 2.5 + b):")
print(f"  - Pure Python (List comprehension + zip): {py_elapsed * 1000:.2f} ms")
print(f"  - Vectorized NumPy (C SIMD Execution)   : {np_elapsed * 1000:.2f} ms")
print(f"  >>> NumPy is {speedup:.1f}x FASTER than Pure Python! <<<")


# ==============================================================================
# 9. REAL-WORLD QUANTITATIVE CASE STUDY: PORTFOLIO RISK & SHARPE RATIO
# ==============================================================================
# Compute annualized expected return, portfolio variance, volatility, and Sharpe
# ratio for a 4-asset portfolio across 252 trading days.

print_separator("9. Real-World Case Study: Vectorized Portfolio Metrics")

rng = np.random.default_rng(seed=101)
trading_days = 252
n_assets = 4
asset_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Synthetic daily returns for 4 assets (mean 0.05% daily, std 1.2% daily)
daily_returns = rng.normal(loc=0.0005, scale=0.012, size=(trading_days, n_assets))

# Portfolio allocations (weights summing to 1.0: 30%, 30%, 20%, 20%)
weights = np.array([0.30, 0.30, 0.20, 0.20])

# 1. Expected Annualized Asset Returns: mean * 252
annualized_returns = daily_returns.mean(axis=0) * trading_days

# 2. Covariance Matrix (252 * cov): shape (4, 4)
cov_matrix = np.cov(daily_returns, rowvar=False) * trading_days

# 3. Portfolio Return: w^T * R
portfolio_expected_return = np.dot(weights, annualized_returns)

# 4. Portfolio Variance: w^T * Cov * w
portfolio_variance = weights @ cov_matrix @ weights
portfolio_volatility = np.sqrt(portfolio_variance)

# 5. Sharpe Ratio (assuming Risk-Free Rate = 3.5% = 0.035)
risk_free_rate = 0.035
sharpe_ratio = (portfolio_expected_return - risk_free_rate) / portfolio_volatility

print(f"Asset Tickers: {asset_tickers}")
print(f"Portfolio Weights: {weights * 100}%")
print("\nIndividual Annualized Metrics:")
for ticker, ret, std in zip(asset_tickers, annualized_returns, np.sqrt(np.diag(cov_matrix))):
    print(f"  {ticker:>5}: Return = {ret * 100:6.2f}%, Volatility = {std * 100:6.2f}%")

print(f"\nAggregated Portfolio Performance:")
print(f"  Expected Annual Return : {portfolio_expected_return * 100:.2f}%")
print(f"  Portfolio Volatility   : {portfolio_volatility * 100:.2f}%")
print(f"  Sharpe Ratio (Rf=3.5%) : {sharpe_ratio:.3f}")
print("================================================================================")
print(" NumPy Numerical Computing Masterclass Completed Successfully! ")
print("================================================================================")
