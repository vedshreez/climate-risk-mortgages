"""
Compares monthly delinquency rates between the Houston MSA (Harvey-exposed,
"treatment") and Dallas-Fort Worth MSA (not Harvey-declared, "control")
loan populations, both Q1+Q2 2017 origination vintages.

Field 4 in the Freddie Mac performance file is Current Loan Delinquency
Status: '0' = current, '1'+ = 30+ days delinquent (numeric buckets),
'R' = REO. We treat anything other than '0' as delinquent.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

COLS = ["loan_id", "period", "upb", "dlq_status"]

def load_perf(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(
            p, sep="|", header=None, usecols=[0, 1, 2, 3],
            names=COLS, dtype=str, low_memory=False,
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def monthly_delinquency_rate(df):
    df = df.copy()
    df["is_delinquent"] = ~df["dlq_status"].isin(["0", "00"])
    monthly = df.groupby("period").agg(
        total_loans=("loan_id", "count"),
        delinquent=("is_delinquent", "sum"),
    )
    monthly["delinquency_rate_pct"] = (monthly["delinquent"] / monthly["total_loans"] * 100).round(3)
    monthly.index = pd.to_datetime(monthly.index, format="%Y%m")
    return monthly.sort_index()

if __name__ == "__main__":
    houston = load_perf(["data/raw/houston_perf_2017Q1.txt", "data/raw/houston_perf_2017Q2.txt"])
    dfw = load_perf(["data/raw/dfw_perf_2017Q1.txt", "data/raw/dfw_perf_2017Q2.txt"])

    houston_monthly = monthly_delinquency_rate(houston)
    dfw_monthly = monthly_delinquency_rate(dfw)

    print("=== Houston monthly delinquency rate ===")
    print(houston_monthly)
    print("\n=== DFW monthly delinquency rate ===")
    print(dfw_monthly)

    comparison = pd.DataFrame({
        "houston_delinquency_pct": houston_monthly["delinquency_rate_pct"],
        "dfw_delinquency_pct": dfw_monthly["delinquency_rate_pct"],
    })
    comparison["gap_pp"] = (comparison["houston_delinquency_pct"] - comparison["dfw_delinquency_pct"]).round(3)
    comparison.to_csv("outputs/houston_vs_dfw_delinquency_comparison.csv")
    print("\n=== Comparison (Houston - DFW gap) ===")
    print(comparison)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(comparison.index, comparison["houston_delinquency_pct"], label="Houston (Harvey-exposed)", color="#c0392b", linewidth=2)
    ax.plot(comparison.index, comparison["dfw_delinquency_pct"], label="Dallas-Fort Worth (control)", color="#2980b9", linewidth=2)
    ax.axvline(datetime(2017, 8, 25), color="black", linestyle="--", linewidth=1, label="Hurricane Harvey landfall (Aug 25, 2017)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delinquency rate (%)")
    ax.set_title("Monthly Loan Delinquency Rate: Houston (Harvey-exposed) vs. Dallas-Fort Worth (control)\nQ1+Q2 2017 Origination Vintages")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/houston_vs_dfw_delinquency.png", dpi=150)
    print("\nSaved outputs/houston_vs_dfw_delinquency.png")
