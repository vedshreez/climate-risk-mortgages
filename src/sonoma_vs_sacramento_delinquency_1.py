"""
Second-disaster validation: Tubbs Fire (Santa Rosa/Sonoma County, CA,
Oct 8-31, 2017) vs. Sacramento (CA, unaffected control), Q1+Q2 2017
origination vintages. Tests whether the Harvey delinquency-spike pattern
generalizes to a different hazard type (wildfire vs. flood) and a
different state/market.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

COLS = ["loan_id", "period", "upb", "dlq_status"]

def load_perf(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p, sep="|", header=None, usecols=[0, 1, 2, 3], names=COLS, dtype=str, low_memory=False)
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
    sonoma = load_perf(["data/raw/sonoma_perf_2017Q1.txt", "data/raw/sonoma_perf_2017Q2.txt"])
    sacramento = load_perf(["data/raw/sacramento_perf_2017Q1.txt", "data/raw/sacramento_perf_2017Q2.txt"])

    sonoma_monthly = monthly_delinquency_rate(sonoma)
    sac_monthly = monthly_delinquency_rate(sacramento)

    comparison = pd.DataFrame({
        "sonoma_delinquency_pct": sonoma_monthly["delinquency_rate_pct"],
        "sacramento_delinquency_pct": sac_monthly["delinquency_rate_pct"],
    })
    comparison["gap_pp"] = (comparison["sonoma_delinquency_pct"] - comparison["sacramento_delinquency_pct"]).round(3)
    comparison.to_csv("outputs/sonoma_vs_sacramento_delinquency_comparison.csv")

    print("=== Tubbs Fire window (Aug 2017 - Aug 2018) ===")
    print(comparison.loc["2017-08-01":"2018-08-01"])

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(comparison.index, comparison["sonoma_delinquency_pct"], label="Sonoma/Santa Rosa (Tubbs Fire-exposed)", color="#d35400", linewidth=2)
    ax.plot(comparison.index, comparison["sacramento_delinquency_pct"], label="Sacramento (control)", color="#2980b9", linewidth=2)
    ax.axvline(datetime(2017, 10, 8), color="black", linestyle="--", linewidth=1, label="Tubbs Fire ignition (Oct 8, 2017)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Delinquency rate (%)")
    ax.set_title("Monthly Loan Delinquency Rate: Sonoma/Santa Rosa (Tubbs Fire) vs. Sacramento (control)\nQ1+Q2 2017 Origination Vintages")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("outputs/sonoma_vs_sacramento_delinquency.png", dpi=150)
    print("\nSaved outputs/sonoma_vs_sacramento_delinquency.png")
