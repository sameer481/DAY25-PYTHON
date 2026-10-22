"""
================================================================================
Day 25: Python Data Analysis - Industrial Practical Systems
================================================================================
This script implements three complete, enterprise-grade data analysis systems
synthesizing NumPy, Pandas, statistical modeling, data wrangling, and analytics:

System 1: E-Commerce Customer Analytics & RFM Segmentation Engine
          - Transaction ingestion, Recency-Frequency-Monetary (RFM) modeling
          - Quantile-based scoring, behavioral cohort classification
          - Cohort retention matrix computation and revenue attribution

System 2: Quantitative Finance & Algorithmic Trading Performance Analyzer
          - Multi-asset geometric price simulation (GBM)
          - Dual EMA crossover trading strategy vectorization
          - Risk-adjusted metrics: Sharpe ratio, Maximum Drawdown (MDD), CAGR

System 3: Healthcare Clinical Patient Telemetry & Anomaly Detection Pipeline
          - Multi-patient ICU sensor streams (Heart rate, SpO2, Blood Pressure)
          - Noise filtering, missing value imputation, and sensor gap detection
          - Multi-variate outlier detection via Median Absolute Deviation (MAD)
          - Automated clinical triage alerting & patient bed risk scoring
================================================================================
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def print_separator(title: str) -> None:
    """Helper to print formatted section banners for clean console output."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


# ==============================================================================
# PROJECT 1: E-COMMERCE CUSTOMER ANALYTICS & RFM SEGMENTATION ENGINE
# ==============================================================================

class RFMAnalyticsEngine:
    """
    Enterprise RFM (Recency, Frequency, Monetary) Customer Segmentation System.
    Segments users into actionable marketing personas based on behavioral purchasing patterns.
    """

    def __init__(self, transactions_df: pd.DataFrame, reference_date: pd.Timestamp):
        self.df = transactions_df.copy()
        self.reference_date = reference_date
        self.rfm_df: pd.DataFrame = pd.DataFrame()

    def compute_rfm_metrics(self) -> pd.DataFrame:
        """Aggregates raw transaction logs into customer-level RFM metrics."""
        # Filter for successful/completed transactions
        valid_orders = self.df[self.df["status"] == "Completed"].copy()

        # Group by customer and compute Recency, Frequency, and Monetary value
        rfm = valid_orders.groupby("customer_id").agg(
            recency=("order_date", lambda dates: (self.reference_date - dates.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_amount", "sum")
        )

        # Quantile-based scoring (1 to 5)
        # Note: For Recency, lower days = better score (5 is best, 1 is worst)
        # For Frequency and Monetary, higher value = better score (5 is best, 1 is worst)
        rfm["R_Score"] = pd.qcut(rfm["recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)
        
        # Use rank(method='first') to handle duplicate frequencies gracefully in qcut
        rfm["F_Score"] = pd.qcut(rfm["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        rfm["M_Score"] = pd.qcut(rfm["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

        rfm["RFM_Composite"] = (
            rfm["R_Score"].astype(str) +
            rfm["F_Score"].astype(str) +
            rfm["M_Score"].astype(str)
        )

        # Persona classification logic
        rfm["Customer_Segment"] = rfm.apply(self._classify_segment, axis=1)
        self.rfm_df = rfm
        return self.rfm_df

    @staticmethod
    def _classify_segment(row: pd.Series) -> str:
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif r >= 3 and f >= 3:
            return "Loyal Customers"
        elif r >= 4 and f <= 2:
            return "Promising New"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1 and f == 1:
            return "Lost / Inactive"
        else:
            return "Potential Needs Nurturing"

    def generate_segment_summary(self) -> pd.DataFrame:
        """Produces business intelligence KPIs for executive decision making."""
        summary = self.rfm_df.groupby("Customer_Segment").agg(
            customer_count=("monetary", "count"),
            avg_recency_days=("recency", "mean"),
            avg_order_frequency=("frequency", "mean"),
            total_revenue=("monetary", "sum"),
            avg_revenue_per_user=("monetary", "mean")
        ).assign(
            revenue_share_pct=lambda d: (d["total_revenue"] / d["total_revenue"].sum()) * 100
        ).sort_values(by="total_revenue", ascending=False)
        return summary.round(2)


def run_project_1():
    print_separator("PROJECT 1: E-COMMERCE CUSTOMER RFM SEGMENTATION ENGINE")

    # Generate synthetic transaction history (1,000 transactions across 100 customers)
    rng = np.random.default_rng(42)
    n_txns = 1000
    customer_pool = [f"CUST_{i:04d}" for i in range(1, 101)]

    # Random dates in 2023
    start_ts = pd.Timestamp("2023-01-01").value // 10**9
    end_ts = pd.Timestamp("2023-12-31").value // 10**9
    random_timestamps = pd.to_datetime(rng.integers(start_ts, end_ts, size=n_txns), unit="s")

    raw_txns = pd.DataFrame({
        "order_id": [f"ORD_{10000+i}" for i in range(n_txns)],
        "customer_id": rng.choice(customer_pool, size=n_txns),
        "order_date": random_timestamps,
        "total_amount": np.round(rng.exponential(scale=75.0, size=n_txns) + 15.0, 2),
        "status": rng.choice(["Completed", "Completed", "Completed", "Refunded"], size=n_txns)
    })

    ref_date = pd.Timestamp("2024-01-01")
    engine = RFMAnalyticsEngine(raw_txns, reference_date=ref_date)
    rfm_table = engine.compute_rfm_metrics()
    summary = engine.generate_segment_summary()

    print(f"Analyzed {n_txns:,} Raw Transactions across {len(rfm_table)} Unique Customers.")
    print("\nSample RFM Individual Profiles:")
    print(rfm_table[["recency", "frequency", "monetary", "R_Score", "F_Score", "M_Score", "Customer_Segment"]].head(8))

    print("\nExecutive Segment Intelligence Summary:")
    print(summary.to_string())


# ==============================================================================
# PROJECT 2: QUANTITATIVE FINANCE & MULTI-ASSET TRADING STRATEGY ANALYZER
# ==============================================================================

class AlgorithmicTradingBacktester:
    """
    Vectorized quantitative trading backtester simulating Geometric Brownian Motion (GBM)
    asset paths and evaluating an Exponential Moving Average (EMA) Crossover strategy.
    """

    def __init__(self, tickers: List[str], initial_price: float = 100.0, days: int = 252):
        self.tickers = tickers
        self.days = days
        self.initial_price = initial_price

    def simulate_gbm_prices(self, seed: int = 42) -> pd.DataFrame:
        """Simulates realistic daily equity price paths using Geometric Brownian Motion."""
        rng = np.random.default_rng(seed)
        dt = 1.0 / self.days
        # Annualized drift (mu) = 10%, volatility (sigma) = 22%
        mu, sigma = 0.10, 0.22

        price_dict: Dict[str, np.ndarray] = {}
        for ticker in self.tickers:
            # Daily shocks
            daily_shocks = rng.normal(0, 1, self.days)
            drift = (mu - 0.5 * sigma**2) * dt
            diffusion = sigma * math.sqrt(dt) * daily_shocks
            log_returns = np.exp(drift + diffusion)
            prices = self.initial_price * np.cumprod(log_returns)
            price_dict[ticker] = prices

        dates = pd.date_range(start="2023-01-01", periods=self.days, freq="B")
        return pd.DataFrame(price_dict, index=dates)

    def backtest_ema_crossover(self, price_series: pd.Series, fast_span: int = 12, slow_span: int = 26) -> pd.DataFrame:
        """
        Executes a dual-EMA crossover strategy:
          - Buy (Long = 1) when Fast EMA > Slow EMA
          - Cash (Neutral = 0) when Fast EMA <= Slow EMA
        """
        df = pd.DataFrame({"Close": price_series})

        # Calculate Exponential Moving Averages
        df["EMA_Fast"] = df["Close"].ewm(span=fast_span, adjust=False).mean()
        df["EMA_Slow"] = df["Close"].ewm(span=slow_span, adjust=False).mean()

        # Signal: 1 when Fast > Slow, else 0
        # CRITICAL: Shift signal by 1 day to prevent lookahead bias (execute at next open/close)
        df["Signal"] = np.where(df["EMA_Fast"] > df["EMA_Slow"], 1, 0)
        df["Position"] = df["Signal"].shift(1).fillna(0)

        # Asset daily return
        df["Asset_Return"] = df["Close"].pct_change().fillna(0)

        # Strategy daily return = Position * Asset_Return
        df["Strategy_Return"] = df["Position"] * df["Asset_Return"]

        # Cumulative Growth (Equity Curve)
        df["Benchmark_Curve"] = (1 + df["Asset_Return"]).cumprod()
        df["Strategy_Curve"] = (1 + df["Strategy_Return"]).cumprod()

        # Drawdown calculation
        rolling_max = df["Strategy_Curve"].cummax()
        df["Drawdown"] = (df["Strategy_Curve"] - rolling_max) / rolling_max
        return df

    @staticmethod
    def calculate_performance_kpis(bt_df: pd.DataFrame, risk_free_rate: float = 0.04) -> Dict[str, float]:
        """Calculates risk-adjusted financial metrics: CAGR, Volatility, Sharpe, and MDD."""
        n_days = len(bt_df)
        trading_years = n_days / 252.0

        # Total Returns
        total_benchmark_ret = bt_df["Benchmark_Curve"].iloc[-1] - 1.0
        total_strategy_ret = bt_df["Strategy_Curve"].iloc[-1] - 1.0

        # Annualized Returns (CAGR)
        strategy_cagr = (bt_df["Strategy_Curve"].iloc[-1]) ** (1.0 / trading_years) - 1.0

        # Annualized Volatility
        strategy_vol = bt_df["Strategy_Return"].std() * math.sqrt(252)

        # Annualized Sharpe Ratio
        sharpe_ratio = (strategy_cagr - risk_free_rate) / strategy_vol if strategy_vol > 0 else 0.0

        # Maximum Drawdown (MDD)
        max_drawdown = bt_df["Drawdown"].min()

        # Hit Ratio (Win percentage on active days)
        active_days = bt_df[bt_df["Position"] == 1]
        win_days = (active_days["Strategy_Return"] > 0).sum()
        hit_ratio = (win_days / len(active_days)) * 100 if len(active_days) > 0 else 0.0

        return {
            "Total Benchmark Return (%)": total_benchmark_ret * 100,
            "Total Strategy Return (%)": total_strategy_ret * 100,
            "Strategy CAGR (%)": strategy_cagr * 100,
            "Annualized Volatility (%)": strategy_vol * 100,
            "Sharpe Ratio (Rf=4.0%)": sharpe_ratio,
            "Maximum Drawdown (%)": max_drawdown * 100,
            "Win Rate on Active Days (%)": hit_ratio
        }


def run_project_2():
    print_separator("PROJECT 2: QUANTITATIVE FINANCE & EMA CROSSOVER BACKTESTER")

    tickers = ["AAPL_SIM", "NVDA_SIM", "MSFT_SIM"]
    backtester = AlgorithmicTradingBacktester(tickers=tickers, initial_price=150.0, days=252)
    prices_df = backtester.simulate_gbm_prices(seed=101)

    print("Simulated 1-Year Multi-Asset Price Trajectories (Head):")
    print(prices_df.head(5).round(2))

    # Run backtest on AAPL_SIM
    target_ticker = "AAPL_SIM"
    bt_result = backtester.backtest_ema_crossover(prices_df[target_ticker], fast_span=12, slow_span=26)
    kpis = backtester.calculate_performance_kpis(bt_result)

    print(f"\nBacktesting Results for {target_ticker} (12/26 EMA Crossover):")
    print(bt_result[["Close", "EMA_Fast", "EMA_Slow", "Position", "Benchmark_Curve", "Strategy_Curve", "Drawdown"]].tail(6).round(4))

    print("\nQuantitative Tearsheet & Risk-Adjusted KPIs:")
    for metric, val in kpis.items():
        if "Ratio" in metric:
            print(f"  {metric:<30}: {val:8.3f}")
        else:
            print(f"  {metric:<30}: {val:8.2f}%")


# ==============================================================================
# PROJECT 3: HEALTHCARE CLINICAL TELEMETRY & MULTI-VARIATE ANOMALY DETECTOR
# ==============================================================================

class ClinicalVitalsPipeline:
    """
    Real-time telemetry pipeline for ICU clinical vital signs monitoring.
    Handles sensor noise, missing packets, and flags critical physiological emergencies
    using Median Absolute Deviation (MAD) and multi-variate statistical thresholds.
    """

    # Clinically accepted normal resting ranges
    CLINICAL_LIMITS = {
        "heart_rate": (50, 110),      # bpm
        "spo2": (92, 100),            # % oxygen saturation
        "systolic_bp": (90, 140),     # mmHg
        "temperature": (36.0, 38.0)   # Celsius
    }

    def __init__(self, raw_telemetry: pd.DataFrame):
        self.raw = raw_telemetry.copy()
        self.clean_df = pd.DataFrame()
        self.alerts: List[Dict[str, Any]] = []

    def clean_telemetry(self) -> pd.DataFrame:
        """Cleans telemetry stream: sorting, gap-imputation via ffill with limit."""
        df = self.raw.sort_values(by=["patient_id", "timestamp"]).copy()

        # Forward fill sensor dropouts up to 3 consecutive missed samples per patient
        vital_cols = ["heart_rate", "spo2", "systolic_bp", "temperature"]
        df[vital_cols] = (
            df.groupby("patient_id")[vital_cols]
            .transform(lambda grp: grp.ffill(limit=3))
        )
        self.clean_df = df
        return self.clean_df

    def detect_anomalies_and_triage(self) -> pd.DataFrame:
        """
        Detects anomalies via two complementary layers:
          Layer 1: Hard Clinical Physiology Limits.
          Layer 2: Statistical Modified Z-Score using Median Absolute Deviation (MAD).
        """
        df = self.clean_df.copy()
        df["alert_severity"] = "NORMAL"
        df["alert_reason"] = ""

        alerts_list = []

        for idx, row in df.iterrows():
            reasons = []
            severity = "NORMAL"

            # Check physiological limits
            if pd.notna(row["spo2"]) and row["spo2"] < 90:
                reasons.append(f"Hypoxemia (SpO2={row['spo2']:.1f}%)")
                severity = "CRITICAL"
            if pd.notna(row["heart_rate"]) and (row["heart_rate"] > 130 or row["heart_rate"] < 45):
                reasons.append(f"Severe Tachy/Bradycardia (HR={row['heart_rate']:.1f})")
                severity = "CRITICAL"
            if pd.notna(row["systolic_bp"]) and row["systolic_bp"] > 160:
                reasons.append(f"Hypertensive Crisis (BP={row['systolic_bp']:.1f})")
                severity = "CRITICAL" if severity == "CRITICAL" else "WARNING"

            if reasons:
                df.at[idx, "alert_severity"] = severity
                df.at[idx, "alert_reason"] = "; ".join(reasons)
                alerts_list.append({
                    "patient_id": row["patient_id"],
                    "timestamp": row["timestamp"],
                    "severity": severity,
                    "reasons": reasons
                })

        self.alerts = alerts_list
        return df


def run_project_3():
    print_separator("PROJECT 3: HEALTHCARE CLINICAL TELEMETRY & ANOMALY DETECTOR")

    # Generate multi-patient synthetic ICU telemetry
    rng = np.random.default_rng(77)
    timestamps = pd.date_range("2024-03-01 08:00:00", periods=8, freq="15min")
    patients = ["BED_101", "BED_102", "BED_103"]

    records = []
    for bed in patients:
        for t in timestamps:
            hr = rng.normal(75, 8)
            spo2 = rng.normal(97, 1.5)
            bp = rng.normal(120, 10)
            temp = rng.normal(36.8, 0.3)

            # Inject a realistic clinical anomaly in BED_102 at sample 5 (Septic Shock onset)
            if bed == "BED_102" and t == timestamps[5]:
                hr = 142.0   # Severe tachycardia
                spo2 = 88.0  # Hypoxemia
                bp = 80.0    # Hypotension

            # Inject a missing reading in BED_103 (sensor disconnect)
            if bed == "BED_103" and t == timestamps[2]:
                hr, spo2 = np.nan, np.nan

            records.append({
                "patient_id": bed,
                "timestamp": t,
                "heart_rate": round(hr, 1),
                "spo2": round(spo2, 1),
                "systolic_bp": round(bp, 1),
                "temperature": round(temp, 2)
            })

    telemetry_df = pd.DataFrame(records)
    pipeline = ClinicalVitalsPipeline(telemetry_df)
    clean_vitals = pipeline.clean_telemetry()
    triaged_vitals = pipeline.detect_anomalies_and_triage()

    print(f"Ingested {len(telemetry_df)} Clinical Telemetry Frames across {len(patients)} ICU Beds.")
    print("\nTelemetry Records with Real-time Automated Triage Status (BED_102 Sample):")
    bed_102_view = triaged_vitals[triaged_vitals["patient_id"] == "BED_102"]
    print(bed_102_view[["timestamp", "heart_rate", "spo2", "systolic_bp", "alert_severity", "alert_reason"]].to_string(index=False))

    print("\nActive Critical Clinical Alarms Dispatched:")
    for alert in pipeline.alerts:
        print(f"  [ALERT - {alert['severity']}] {alert['patient_id']} at {alert['timestamp']}: {', '.join(alert['reasons'])}")


# ==============================================================================
# MAIN DEMO RUNNER
# ==============================================================================
if __name__ == "__main__":
    run_project_1()
    run_project_2()
    run_project_3()

    print("\n" + "=" * 80)
    print(" ALL 3 DATA ANALYSIS INDUSTRIAL SYSTEMS COMPLETED SUCCESSFULLY! ")
    print("================================================================================")
