# Quantifying Climate Risk in Residential Mortgages

Empirically testing the lead-lag relationship between climate disaster exposure and residential mortgage delinquency, using real Freddie Mac loan-level performance data, FEMA disaster declarations, and two independent disaster events across different hazard types.

Vedashree Teli | NC State University

---

## Introduction

Climate change is increasingly relevant to how lenders and GSEs (government-sponsored enterprises) think about mortgage risk in coastal zones, floodplains, and wildfire-prone regions. This project doesn't just look at losses *after* a disaster hits — it directly tests whether real loan performance data shows a measurable, dateable divergence between disaster-exposed loans and matched, unaffected control loans, using two separate real-world events: **Hurricane Harvey** (Houston, TX, Aug 2017) and the **Tubbs Fire** (Sonoma/Napa, CA, Oct 2017).

## Motivation

Research shows climate-prone mortgages are increasingly being securitized, which makes it a genuine GSE-level risk question, not just an academic one. Most climate-risk mortgage projects rely on synthetic scenario modeling; this project instead uses actual historical loan performance around two real, independently verifiable disasters to see whether the effect is visible in the data itself — and whether it generalizes across hazard types (flood vs. wildfire) and geographies (Texas vs. California).

## Data Sources

- **Freddie Mac Single-Family Loan-Level Dataset (SFLLD)** — free, registration-required, via [freddiemac.com/research/datasets/sf-loanlevel-dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset). Origination + full monthly performance history, 2017Q1–Q2 vintages.
- **FEMA Disaster Declarations** — fully public, no registration, via the [OpenFEMA API](https://www.fema.gov/about/openfema/api). Confirms Harris County (Houston) as a declared disaster area under DR-4332-TX (Hurricane Harvey).
- **Public knowledge of disaster dates/locations** — Hurricane Harvey landfall (Aug 25, 2017); Tubbs Fire ignition (Oct 8, 2017, Sonoma County).

## Why Two Disasters, Two Hazard Types

A single case study can't distinguish "this disaster affected mortgage performance" from "this specific event happened to coincide with something else." Testing the same hypothesis against a **second, independent disaster of a completely different hazard type** (wildfire vs. hurricane flooding) in a **different state** is a genuine generalization test, not just a repeated example.

## Methodology

### 1. Treatment/control design
For each disaster, loans are split into:
- **Treatment**: loans in the disaster-declared metro area, originated *before* the disaster (so they're seasoned, existing loans at the time of the shock — not loans originated in response to it).
- **Control**: loans in a nearby, unaffected metro area in the same state, same origination vintage — holding state-level economic conditions roughly constant.

| Disaster | Treatment | Control |
|---|---|---|
| Hurricane Harvey (Aug 2017) | Houston MSA (26420) — 12,595 loans | Dallas-Fort Worth MSA (19124) — 12,583 loans |
| Tubbs Fire (Oct 2017) | Sonoma+Napa zip3 (954, 945) — 7,260 loans | Sacramento+Stockton zip3 (958, 956) — 6,441 loans |

All loans are Q1–Q2 2017 originations, so they're already seasoned by the time each disaster hits.

### 2. Descriptive validation: monthly delinquency rate comparison
For each population, computes the % of loans reporting any delinquency status each month, plotted treatment vs. control over time, with the disaster date marked. The test: do the two lines track together *before* the disaster and diverge *after*?

### 3. Loan-level model: 90+ day delinquency prediction
An XGBoost classifier predicts whether a loan reached 90+ day delinquency during the post-disaster window, using **only origination-time features** (credit score, DTI, LTV/CLTV, loan purpose, occupancy, etc. — no post-event data, avoiding leakage) plus a `disaster_exposed` flag and explicit vulnerability interaction terms (`exposed_x_low_credit`, `exposed_x_high_dti`, `exposed_x_high_ltv`).

**Metric choice matters here**: with a rare positive class (delinquency events are a small % of loans), ROC-AUC can look strong while barely beating a naive baseline. PR-AUC (precision-recall AUC) and the KS statistic are reported alongside ROC-AUC specifically to avoid overstating model quality on imbalanced data.

### 4. SHAP analysis
Identifies which features actually drive the model's predictions — specifically checking whether disaster exposure interacts with financial fragility (e.g., low credit score) rather than acting as an independent risk factor.

## Results

### Hurricane Harvey — strong, clean validation

| Month | Houston Delinquency | Dallas-Fort Worth Delinquency | Gap |
|---|---|---|---|
| Aug 2017 (pre-storm) | 0.54% | 0.31% | 0.22pp |
| Sep 2017 | 4.39% | 0.46% | 3.93pp |
| Oct 2017 (peak) | 5.58% | 0.44% | 5.13pp |

Houston and Dallas track almost identically before the storm, then Houston's delinquency rate jumps roughly **8-10x** the month Harvey hits, peaking at over 12x Dallas's rate a month later, before decaying over the following year.

**Loan-level model**: ROC-AUC 0.833, PR-AUC 0.065 (vs. 0.014 baseline — a genuine ~4.6x lift), KS 0.554. **Top SHAP feature: `exposed_x_low_credit`** — the interaction term, not disaster exposure or credit score alone — confirming that climate exposure amplifies existing financial fragility rather than acting independently. See `outputs/shap_summary_plot.png`.

### Tubbs Fire — pattern generalizes, but smaller and slower

| Month | Sonoma Delinquency | Sacramento Delinquency | Gap |
|---|---|---|---|
| Aug 2017 (pre-fire) | 0.16% | 0.41% | -0.25pp |
| Oct 2017 (ignition month) | 0.97% | 0.50% | 0.47pp |
| Dec 2017 (peak) | 2.32% | 0.38% | 1.93pp |

The same directional pattern holds — a real divergence starting right around the disaster date — but roughly **half the magnitude** of Harvey, with a **slower onset** (peaking 2 months after ignition vs. 1 month after Harvey's landfall). This is a genuinely interesting finding on its own: wildfire damage-to-delinquency transmission appears slower than flood damage, consistent with the more spatially concentrated, displacement-driven nature of wildfire loss vs. the immediate, broad flooding from a hurricane.

**Loan-level model — an honest limitation, not a clean replication**: with only 12-23 positive delinquency events in this smaller population, the loan-level model does not reliably replicate the Harvey findings (ROC-AUC ranged 0.39–0.59 across geographic specifications, PR-AUC barely above baseline). Widening the geography (adding Napa County) increased the sample and improved the model modestly, but **diluted the effect size** (exposed-group delinquency dropped from 0.85% to 0.23%), most likely because the broader zip3 code mixes in some unaffected areas (Freddie Mac only provides 3-digit ZIP codes for privacy, so finer geographic precision isn't available in this dataset). **The descriptive validation clearly generalizes; the loan-level model requires a larger disaster or finer geographic data than this specific case provides.**

## Repository Structure

```
data/raw/          — origination files (small, included); performance files excluded (see below)
src/                — all analysis scripts
outputs/            — charts, comparison tables, model results, SHAP plots
notes/              — (future) full write-up
```

## Reproducing the Data

The raw monthly performance files (~190MB combined) are excluded from this repo via `.gitignore` due to size. To regenerate them:

1. Register (free) at [freddiemac.com/research/datasets/sf-loanlevel-dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset)
2. Download the **Standard Dataset** for **2017Q1** and **2017Q2** (origination period)
3. Each quarter's zip contains `historical_data_YYYYQ#.txt` (origination) and `historical_data_time_YYYYQ#.txt` (monthly performance)
4. Filter to the relevant geography using the scripts in `src/` (property state = field 17, MSA = field 5, postal code = field 19 in the origination file)

## Key Takeaways

1. **A real, dateable divergence in mortgage performance is directly visible around both disasters**, using nothing but a treatment/control comparison — no complex modeling required to see the core effect.
2. **The effect generalizes across hazard types** (hurricane flooding vs. wildfire) but with meaningfully different magnitude and timing — hurricanes produce a sharp, fast spike; wildfires produce a smaller, slower-building one.
3. **Climate exposure interacts with financial fragility** rather than acting as an independent risk factor — the strongest loan-level predictor in the Harvey model was the interaction between disaster exposure and low credit score, not either factor alone.
4. **Loan-level modeling of rare events needs real statistical power** — a smaller disaster and coarser geographic data (zip3 vs. exact address) can produce an honest null result even when the underlying phenomenon is real, and that's worth reporting plainly rather than engineering around it.

## References

- Freddie Mac Single-Family Loan-Level Dataset, standard dataset, 2017Q1–Q2 vintages.
- FEMA OpenFEMA API, Disaster Declarations Summaries (DR-4332-TX, Hurricane Harvey).
- Public reporting on Hurricane Harvey (landfall Aug 25, 2017) and the Tubbs Fire (ignition Oct 8, 2017, part of the October 2017 Northern California firestorm).
