"""
Second-disaster validation model: predicts 90+ day delinquency during the
Tubbs Fire window (Oct 2017 - Feb 2018, reflecting the slower onset seen in
the descriptive comparison) using ONLY origination-time features plus a
disaster-exposure flag (Sonoma/Santa Rosa zip3=954 = 1, Sacramento zip3=958
control = 0). Mirrors build_pd_model.py exactly so results are comparable.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

ORIG_COLS = [
    "credit_score", "first_payment_date", "first_time_homebuyer", "maturity_date",
    "msa", "mi_pct", "num_units", "occupancy_status", "cltv", "dti",
    "orig_upb", "ltv", "orig_interest_rate", "channel", "ppm_flag",
    "amortization_type", "property_state", "property_type", "postal_code",
    "loan_seq_num", "loan_purpose", "orig_loan_term", "num_borrowers",
    "seller_name", "servicer_name", "super_conforming_flag",
    "pre_harp_lsn", "program_indicator", "harp_indicator",
    "property_valuation_method", "interest_only_flag", "mi_cancellation",
]

def load_origination(path, disaster_exposed):
    df = pd.read_csv(path, sep="|", header=None, names=ORIG_COLS, dtype=str, low_memory=False)
    df["disaster_exposed"] = disaster_exposed
    return df

def load_performance_target(paths, window_start="2017-10", window_end="2018-02"):
    frames = []
    for p in paths:
        d = pd.read_csv(p, sep="|", header=None, usecols=[0, 1, 3],
                         names=["loan_seq_num", "period", "dlq_status"], dtype=str, low_memory=False)
        frames.append(d)
    perf = pd.concat(frames, ignore_index=True)
    perf["period_dt"] = pd.to_datetime(perf["period"], format="%Y%m")
    in_window = perf[(perf["period_dt"] >= window_start) & (perf["period_dt"] <= window_end)].copy()

    def is_90plus(x):
        if x in ("0", "00", "0 ", None) or pd.isna(x):
            return False
        if x == "R":
            return True
        try:
            return int(x) >= 3
        except ValueError:
            return False

    in_window["is_90plus"] = in_window["dlq_status"].apply(is_90plus)
    target = in_window.groupby("loan_seq_num")["is_90plus"].max().astype(int)
    return target

if __name__ == "__main__":
    print("Loading origination data...")
    sonoma_orig = load_origination("data/raw/sonoma_orig_2017Q1Q2.txt", disaster_exposed=1)
    sac_orig = load_origination("data/raw/sacramento_orig_2017Q1Q2.txt", disaster_exposed=0)
    orig = pd.concat([sonoma_orig, sac_orig], ignore_index=True)
    print(f"Total loans: {len(orig)} (Sonoma: {len(sonoma_orig)}, Sacramento: {len(sac_orig)})")

    print("Building target variable from performance data (Oct 2017-Feb 2018 window)...")
    target = load_performance_target([
        "data/raw/sonoma_perf_2017Q1.txt", "data/raw/sonoma_perf_2017Q2.txt",
        "data/raw/sacramento_perf_2017Q1.txt", "data/raw/sacramento_perf_2017Q2.txt",
    ])

    orig = orig.set_index("loan_seq_num")
    orig["target_90plus_dlq"] = target.reindex(orig.index).fillna(0).astype(int)

    print(f"\nOverall 90+ day delinquency rate: {orig['target_90plus_dlq'].mean()*100:.2f}%")
    print(f"Sonoma rate: {orig[orig.disaster_exposed==1]['target_90plus_dlq'].mean()*100:.2f}%")
    print(f"Sacramento rate: {orig[orig.disaster_exposed==0]['target_90plus_dlq'].mean()*100:.2f}%")

    numeric_cols = ["credit_score", "cltv", "dti", "orig_upb", "ltv", "orig_interest_rate", "num_units", "num_borrowers"]
    for c in numeric_cols:
        orig[c] = pd.to_numeric(orig[c], errors="coerce")

    orig.loc[orig["credit_score"] >= 9999, "credit_score"] = np.nan
    orig.loc[orig["dti"] >= 999, "dti"] = np.nan
    orig.loc[orig["cltv"] >= 999, "cltv"] = np.nan
    orig.loc[orig["ltv"] >= 999, "ltv"] = np.nan

    orig["exposed_x_high_ltv"] = orig["disaster_exposed"] * (orig["ltv"].fillna(orig["ltv"].median()) / 100)
    orig["exposed_x_low_credit"] = orig["disaster_exposed"] * ((850 - orig["credit_score"].fillna(orig["credit_score"].median())) / 850)
    orig["exposed_x_high_dti"] = orig["disaster_exposed"] * (orig["dti"].fillna(orig["dti"].median()) / 100)

    categorical_cols = ["first_time_homebuyer", "occupancy_status", "channel", "amortization_type",
                         "property_type", "loan_purpose"]
    for c in categorical_cols:
        orig[c] = orig[c].fillna("MISSING").astype("category")

    feature_cols = numeric_cols + ["disaster_exposed", "exposed_x_high_ltv", "exposed_x_low_credit",
                                     "exposed_x_high_dti"] + categorical_cols

    X = orig[feature_cols].copy()
    for c in categorical_cols:
        X[c] = X[c].cat.codes
    y = orig["target_90plus_dlq"]

    print(f"\nPositive count: {y.sum()} / {len(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train positive rate: {y_train.mean()*100:.2f}%, Test positive rate: {y_test.mean()*100:.2f}%")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)
    ks = max(tpr - fpr)

    print(f"\n=== Model Performance (Test Set) ===")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}  (baseline = {y_test.mean():.3f})")
    print(f"KS statistic: {ks:.3f}")

    with open("outputs/sonoma_model_performance.txt", "w") as f:
        f.write(f"ROC-AUC: {roc_auc:.3f}\nPR-AUC: {pr_auc:.3f} (baseline={y_test.mean():.3f})\nKS: {ks:.3f}\n")

    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    mean_abs_shap = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print("\n=== Top 10 features by mean |SHAP value| ===")
    print(mean_abs_shap.head(10))
    mean_abs_shap.to_csv("outputs/sonoma_shap_feature_importance.csv")

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("outputs/sonoma_shap_summary_plot.png", dpi=150, bbox_inches="tight")
    print("\nSaved outputs/sonoma_shap_summary_plot.png")
