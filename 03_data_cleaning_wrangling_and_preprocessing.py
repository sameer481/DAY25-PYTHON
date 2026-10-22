"""
================================================================================
Day 25: Python Data Analysis - Data Cleaning, Wrangling & Preprocessing
================================================================================
Data cleaning and wrangling account for up to 80% of a data analyst or machine
learning engineer's day-to-day workflow. Real-world data is inherently noisy,
incomplete, inconsistent, and often structurally misaligned for statistical models.

Theoretical Background:
Missing data falls into three formal statistical categories (Donald Rubin, 1976):
  1. MCAR (Missing Completely at Random): Missingness is independent of both
     observed and unobserved data (e.g., random sensor communication drop).
  2. MAR (Missing at Random): Missingness systematically depends on other
     observed variables (e.g., young people omitting income).
  3. MNAR (Missing Not at Random): Missingness depends on the unobserved value
     itself (e.g., people with extremely high wealth refusing wealth disclosures).

Key Topics Covered in this Script:
1. Missing Data Auditing & Statistical Mechanisms (MCAR / MAR / MNAR)
2. Strategic Imputation: Mean, Median, Mode, Forward/Backward Fill & Group Imputation
3. Duplicate Detection & Precision Deduplication Strategies
4. Outlier Detection & Robust Treatment: Z-Scores vs. Interquartile Range (IQR)
5. Type Casting, Schema Sanitization & Regex-Driven Dirty String Normalization
6. Reshaping Tabular Data: `.pivot_table()`, `.melt()`, `.stack()` & `.unstack()`
7. Contingency Tables & Frequency Analysis with `pd.crosstab()`
8. Real-World Production Pipeline: Automated End-to-End Raw Transaction Sanitizer
================================================================================
"""

import io
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def print_separator(title: str) -> None:
    """Helper to print formatted section banners for clean console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# 1. MISSING DATA AUDITING & STATISTICAL PROFILING
# ==============================================================================
# Auditing percentage of missing values per column and understanding patterns.

print_separator("1. Missing Data Auditing & Null Profiling")

dirty_data = '''id,full_name,age,gender,income,city,signup_channel
U101,  John Doe ,28,Male,"$85,000",New York,Organic
U102,Jane Smith,34,F,"$112,000",San Francisco,Paid Ad
U103, Bob Miller ,NaN,m,"$54,000",new york,Referral
U104,Alice Brown,45,Female,?,Chicago,Organic
U105,Charlie Black,29,M,"$72,000",SAN FRANCISCO,Organic
U106,David Wilson,NaN,Male,"$68,000",New York,
U107,Eva Green,52,F,"$145,000",,Paid Ad
U108,Frank White,31,Male,N/A,Chicago,Referral
U109,Bob Miller,NaN,m,"$54,000",new york,Referral
U110,Grace Hall,24,Female,"$48,000",Chicago,Organic
'''

# Ingest and specify custom missing value sentinels ('?', 'N/A')
df_raw = pd.read_csv(
    io.StringIO(dirty_data),
    na_values=["?", "N/A", ""]
)

print("Raw Ingested Dataset with Custom Missing Sentinels:")
print(df_raw)

# Calculate missingness audit
missing_summary = pd.DataFrame({
    "Missing_Count": df_raw.isna().sum(),
    "Missing_Percent": (df_raw.isna().mean() * 100).round(2),
    "Dtype": df_raw.dtypes
})
print("\nMissingness Audit Table:")
print(missing_summary)


# ==============================================================================
# 2. STRATEGIC IMPUTATION: MEAN, MEDIAN, FORWARD FILL & GROUP MEDIANS
# ==============================================================================
# Imputation Strategies:
#   - Dropping (`dropna`): Good when missingness is tiny (<2%) and MCAR.
#   - Global Mean/Median: Biases variance downward.
#   - Group-conditional Median: Imputes values conditioned on subgroup characteristics.

print_separator("2. Strategic Missing Value Imputation")

df_impute = df_raw.copy()

# Cleaning income column first so it's numeric for imputation
df_impute["income"] = (
    df_impute["income"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
df_impute["income"] = pd.to_numeric(df_impute["income"], errors="coerce")

print(f"Age before imputation: {df_impute['age'].tolist()}")

# 1. Global median imputation for age
global_median_age = df_impute["age"].median()
print(f"Global Median Age: {global_median_age}")

# 2. Forward fill and backward fill demonstration for time-series / sequential logs
ts_series = pd.Series([10.0, np.nan, np.nan, 25.0, np.nan, 40.0])
print(f"\nSequential Series with Gaps: {ts_series.tolist()}")
print(f"Forward Filled (.ffill())  : {ts_series.ffill().tolist()}")
print(f"Backward Filled (.bfill()) : {ts_series.bfill().tolist()}")

# 3. Advanced: Group-conditional median income imputation based on gender/channel
print("\nGroup-conditional Median Imputation:")
income_by_gender = df_impute.groupby("gender")["income"].transform("median")
df_impute["income_imputed"] = df_impute["income"].fillna(income_by_gender).fillna(df_impute["income"].median())
print(df_impute[["id", "gender", "income", "income_imputed"]])


# ==============================================================================
# 3. DUPLICATE DETECTION & DEDUPLICATION STRATEGIES
# ==============================================================================
# Full-row duplicates vs. semantic duplicates (same user/customer).

print_separator("3. Duplicate Detection & Strategic Deduplication")

print(f"Total Rows Before Deduplication: {len(df_raw)}")

# Check duplicate rows across all columns
full_dups = df_raw.duplicated(keep=False)
print(f"Full duplicate rows found: {full_dups.sum()}")

# Semantic duplicate: Same name and income
subset_dups = df_raw.duplicated(subset=["full_name", "income"], keep="first")
print(f"Subset duplicate rows (name + income): {subset_dups.sum()}")

# Deduplicate keeping the first occurrence
df_dedup = df_raw.drop_duplicates(subset=["full_name", "income"], keep="first").copy()
print(f"Rows remaining after dropping duplicates: {len(df_dedup)}")


# ==============================================================================
# 4. OUTLIER DETECTION & TREATMENT: Z-SCORE VS. IQR FENCE
# ==============================================================================
# Outliers distort statistical estimators like mean and standard deviation.
#   1. Z-Score: assumes Gaussian normal distribution. Flags values where |Z| > 3.
#   2. IQR (Interquartile Range) Fence: robust to non-normal distributions.
#      Lower Fence = Q1 - 1.5 * IQR
#      Upper Fence = Q3 + 1.5 * IQR

print_separator("4. Outlier Detection: Z-Score vs. IQR Fence Rule")

rng = np.random.default_rng(42)
# Generate 100 normal observations + 4 extreme anomalies
sample_salaries = rng.normal(loc=75_000, scale=12_000, size=100)
outliers = np.array([250_000, 310_000, 8_000, 15_000])
salary_data = pd.Series(np.concatenate([sample_salaries, outliers]), name="salary")

# Method 1: Z-Score Detection
z_scores = (salary_data - salary_data.mean()) / salary_data.std()
z_outliers = salary_data[z_scores.abs() > 3]
print(f"1. Z-Score Outliers (|Z| > 3):\n{z_outliers.values}")

# Method 2: IQR Fence Rule (Tukey's Fences)
q1 = salary_data.quantile(0.25)
q3 = salary_data.quantile(0.75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

iqr_outliers = salary_data[(salary_data < lower_fence) | (salary_data > upper_fence)]
print(f"\n2. IQR Fence Rule:")
print(f"   Q1 = ${q1:,.2f}, Q3 = ${q3:,.2f}, IQR = ${iqr:,.2f}")
print(f"   Lower Fence = ${lower_fence:,.2f}, Upper Fence = ${upper_fence:,.2f}")
print(f"   Outliers Detected by IQR ({len(iqr_outliers)} items): {iqr_outliers.values}")

# Treatment: Winsorization (Capping at the fences rather than discarding data)
capped_salaries = salary_data.clip(lower=lower_fence, upper=upper_fence)
print(f"\nWinsorized (Capped) Min: ${capped_salaries.min():,.2f}")
print(f"Winsorized (Capped) Max: ${capped_salaries.max():,.2f}")


# ==============================================================================
# 5. STRING NORMALIZATION, CATEGORY STANDARDIZATION & REGEX EXTRACTION
# ==============================================================================
# Dirty categorical strings ("m", "Male", "MALE", "F", "Female") and messy city names.

print_separator("5. Category Normalization & Regex Pattern Parsing")

df_clean = df_dedup.copy()

# 1. Clean and normalize full_name
df_clean["full_name"] = df_clean["full_name"].str.strip().str.title()

# 2. Standardize Gender with mapping dictionary
gender_map = {
    "male": "Male", "m": "Male",
    "female": "Female", "f": "Female"
}
df_clean["gender_clean"] = (
    df_clean["gender"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map(gender_map)
    .fillna("Unknown")
)

# 3. Standardize City (Title Case)
df_clean["city_clean"] = df_clean["city"].astype(str).str.strip().str.title().replace("Nan", "Unknown")

print("Standardized Demographics:")
print(df_clean[["full_name", "gender", "gender_clean", "city", "city_clean"]])

# 4. Regex Extraction from unstructured log strings
log_entries = pd.Series([
    "GET /api/v1/products?cat=electronics HTTP/1.1 200 1420ms",
    "POST /api/v1/checkout HTTP/1.1 500 8500ms",
    "GET /api/v1/users/U104 HTTP/1.1 404 120ms"
], name="raw_log")

log_pattern = r"(?P<method>GET|POST|PUT|DELETE)\s+(?P<endpoint>\S+)\s+\S+\s+(?P<status>\d{3})\s+(?P<latency_ms>\d+)ms"
parsed_logs = log_entries.str.extract(log_pattern)
parsed_logs["status"] = parsed_logs["status"].astype(int)
parsed_logs["latency_ms"] = parsed_logs["latency_ms"].astype(int)

print("\nExtracted Structured Data from Messy Server Logs:")
print(parsed_logs)


# ==============================================================================
# 6. RESHAPING DATA: PIVOT TABLES, MELT, STACK & UNSTACK
# ==============================================================================
# Tabular data comes in two fundamental formats:
#   - Long Format (tidy data): One observation per row (best for seaborn, analysis, DBs).
#   - Wide Format: Variables spread across columns (best for executive reporting tables).

print_separator("6. Reshaping: Long vs. Wide (pivot_table & melt)")

sales_records = pd.DataFrame({
    "Store": ["Downtown", "Downtown", "Suburbs", "Suburbs", "Downtown", "Suburbs"],
    "Quarter": ["Q1", "Q2", "Q1", "Q2", "Q1", "Q2"],
    "Product": ["Laptops", "Laptops", "Laptops", "Laptops", "Phones", "Phones"],
    "Revenue": [50_000, 62_000, 38_000, 41_000, 80_000, 75_000],
    "Units_Sold": [40, 50, 30, 32, 100, 95]
})

print("Original Long Format Sales Records:")
print(sales_records)

# 1. Pivot Table: Store & Product vs. Quarter Revenue
wide_pivot = sales_records.pivot_table(
    index=["Store", "Product"],
    columns="Quarter",
    values="Revenue",
    aggfunc="sum",
    fill_value=0
)
print("\nPivoted Table (Wide Format - Revenue by Quarter):")
print(wide_pivot)

# 2. Melt: Convert Wide Format Back into Long Tidy Format
reset_pivot = wide_pivot.reset_index()
melted_df = pd.melt(
    reset_pivot,
    id_vars=["Store", "Product"],
    value_vars=["Q1", "Q2"],
    var_name="Fiscal_Quarter",
    value_name="Total_Revenue"
)
print("\nMelted DataFrame (Back to Long Format):")
print(melted_df)


# ==============================================================================
# 7. CONTINGENCY TABLES & CROSS-TABULATIONS (pd.crosstab)
# ==============================================================================
# Computes a simple cross tabulation of two or more categorical factors.

print_separator("7. Contingency Tables & Frequency Analysis (pd.crosstab)")

survey_df = pd.DataFrame({
    "Subscription": ["Free", "Premium", "Free", "Premium", "Enterprise", "Free", "Premium", "Enterprise", "Free", "Premium"],
    "Satisfaction": ["Satisfied", "Satisfied", "Neutral", "Dissatisfied", "Satisfied", "Dissatisfied", "Satisfied", "Satisfied", "Neutral", "Satisfied"]
})

# Cross-tabulation with row percentages (margins=True adds subtotals)
cross_tab_counts = pd.crosstab(survey_df["Subscription"], survey_df["Satisfaction"], margins=True)
cross_tab_pct = pd.crosstab(survey_df["Subscription"], survey_df["Satisfaction"], normalize="index") * 100

print("Satisfaction Counts Across Subscription Tiers:")
print(cross_tab_counts)
print("\nSatisfaction Row Percentages (% within each tier):")
print(cross_tab_pct.round(1))


# ==============================================================================
# 8. PRODUCTION PIPELINE: AUTOMATED DIRTY TRANSACTION DATA SANITIZER
# ==============================================================================
print_separator("8. Production Pipeline: End-to-End Raw Transaction Sanitizer")

class RawDataSanitizer:
    """Robust production-grade data cleansing pipeline for transactional feeds."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.audit_log: List[str] = []

    def clean_missing_and_duplicates(self) -> "RawDataSanitizer":
        initial_len = len(self.df)
        self.df.drop_duplicates(inplace=True)
        dropped_dups = initial_len - len(self.df)
        self.audit_log.append(f"Dropped {dropped_dups} duplicate records.")
        return self

    def sanitize_currency(self, columns: List[str]) -> "RawDataSanitizer":
        for col in columns:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.replace(r"[^\d.]", "", regex=True)
                )
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                self.audit_log.append(f"Sanitized currency column '{col}', imputed nulls with median ({median_val:.2f}).")
        return self

    def filter_outliers_iqr(self, column: str, multiplier: float = 1.5) -> "RawDataSanitizer":
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - multiplier * iqr, q3 + multiplier * iqr
        outlier_count = ((self.df[column] < low) | (self.df[column] > high)).sum()
        # Cap rather than discard
        self.df[column] = self.df[column].clip(lower=low, upper=high)
        self.audit_log.append(f"Winsorized {outlier_count} outliers in '{column}' to range [{low:.2f}, {high:.2f}].")
        return self

    def get_result(self) -> Tuple[pd.DataFrame, List[str]]:
        return self.df, self.audit_log


# Run pipeline on synthetic raw batch
dirty_batch = pd.DataFrame({
    "trans_id": ["T1", "T2", "T3", "T4", "T4", "T5", "T6"],
    "amount": ["$145.20", "€95.00", "$99999.00", "$320.50", "$320.50", "N/A", "$12.00"],
    "status": ["Completed", "Pending", "Completed", "Refunded", "Refunded", "Completed", "Completed"]
})

sanitizer = RawDataSanitizer(dirty_batch)
clean_df, logs = (
    sanitizer
    .clean_missing_and_duplicates()
    .sanitize_currency(["amount"])
    .filter_outliers_iqr("amount")
    .get_result()
)

print("Pipeline Execution Audit Logs:")
for log in logs:
    print(f"  [AUDIT] {log}")

print("\nFinal Cleaned & Sanitized DataFrame:")
print(clean_df)

print("================================================================================")
print(" Data Cleaning, Wrangling & Preprocessing Masterclass Completed! ")
print("================================================================================")
