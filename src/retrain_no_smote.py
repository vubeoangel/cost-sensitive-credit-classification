"""
Retrain LightGBM WITHOUT SMOTE (using scale_pos_weight instead) and compare the
cost-optimal decision threshold against the SMOTE-trained baseline.

Purpose: diagnose whether the empirical optimum t=0.01 is an artefact of SMOTE
rebalancing the training prior, or a property of the data itself.

Usage (from the project root or the test/ folder):
    python retrain_no_smote.py

Outputs: console report + no_smote_threshold_results.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, "..", "data", "train.csv"))
OUT_CSV = os.path.normpath(os.path.join(HERE, "..", "results", "no_smote_threshold_results.csv"))

# ─── Cost constants (from journal §2.1.5) ─────────────────────────────────────
COST_FN = 1922.58    # approving a defaulter
COST_FP = 253.74     # rejecting a good customer
COST_RATIO = COST_FN / COST_FP

# ─── 1. Load + preprocess (mirrors the notebook) ──────────────────────────────
print("─" * 70)
print("Loading and preprocessing data...")
df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

# Drop irrelevant id columns
df = df.drop(columns=["customer_id", "name"])

# Drop invalid employment records (same rule as notebook)
df = df[df["no_of_days_employed"] <= 36500].copy()

# Binary categorical: map to 0/1 (fill nulls with mode first)
for col in ["gender", "owns_car", "owns_house"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
# Unify the rogue 'XNA' gender value with the mode ('F')
df["gender"] = df["gender"].replace("XNA", df["gender"].mode().iloc[0])
binary_maps = {"gender": {"M": 1, "F": 0},
               "owns_car": {"Y": 1, "N": 0},
               "owns_house": {"Y": 1, "N": 0}}
for col, mapping in binary_maps.items():
    df[col] = df[col].map(mapping)

# Numeric null imputation (median) — matches notebook cell 23
num_feats = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_feats:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].median())

# Engineered features (cell 52)
df["debt_to_income_ratio"] = df["yearly_debt_payments"] / df["net_yearly_income"].replace(0, np.nan)
df["children_family_ratio"] = df["no_of_children"] / df["total_family_members"].replace(0, np.nan)
df["income_per_family_member"] = df["net_yearly_income"] / df["total_family_members"].replace(0, np.nan)
for col in ["debt_to_income_ratio", "children_family_ratio", "income_per_family_member"]:
    df[col] = df[col].fillna(df[col].median())

# Drop redundant columns (cell 59) and one-hot encode occupation (cell 60)
df = df.drop(columns=["credit_limit", "total_family_members", "owns_house"])
df = pd.get_dummies(df, columns=["occupation_type"], drop_first=True, dtype=int)

target = "credit_card_default"
X = df.drop(columns=[target])
y = df[target]
print(f"Post-preprocess shape: X={X.shape}, positive rate={y.mean():.4f}")

# ─── 2. Train / test split + scaling ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")
print(f"Train positive rate: {y_train.mean():.4f}")


# ─── 3. Helpers ───────────────────────────────────────────────────────────────
def threshold_sweep(y_true, y_prob, thresholds):
    """Return array of total costs over threshold grid."""
    costs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        costs.append(fn * COST_FN + fp * COST_FP)
    return np.asarray(costs)


def evaluate_at(threshold, y_true, y_prob, label):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cost = fn * COST_FN + fp * COST_FP
    return {
        "variant": label,
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "FN": int(fn),
        "FP": int(fp),
        "total_cost": round(float(cost), 2),
    }


THRESHOLDS = np.linspace(0.001, 0.99, 1000)   # finer low end to catch true optimum

# Shared LightGBM hyperparameters (best from the notebook's GridSearch)
BASE_PARAMS = dict(
    learning_rate=0.1,
    n_estimators=200,
    max_depth=-1,
    num_leaves=50,
    subsample=0.8,
    random_state=42,
    verbose=-1,
)

# ─── 4A. Variant 1 — Pure imbalanced training (NO SMOTE, NO reweighting) ─────
print("\n" + "─" * 70)
print("Variant 1: LightGBM on RAW imbalanced data (no SMOTE, no reweighting)")
print("─" * 70)
lgbm_raw = LGBMClassifier(**BASE_PARAMS)
lgbm_raw.fit(X_train_s, y_train)
prob_raw = lgbm_raw.predict_proba(X_test_s)[:, 1]
costs_raw = threshold_sweep(y_test.values, prob_raw, THRESHOLDS)
t_opt_raw = float(THRESHOLDS[int(np.argmin(costs_raw))])
print(f"Empirical optimum threshold (raw)  : {t_opt_raw:.4f}")
print(f"Theoretical optimum t*=CFP/(CFN+CFP): {COST_FP/(COST_FP+COST_FN):.4f}")

# ─── 4B. Variant 2 — class-weighted loss (scale_pos_weight) ──────────────────
print("\n" + "─" * 70)
print("Variant 2: LightGBM with scale_pos_weight (no SMOTE, weighted loss)")
print("─" * 70)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.2f}")
lgbm_cw = LGBMClassifier(scale_pos_weight=scale_pos_weight, **BASE_PARAMS)
lgbm_cw.fit(X_train_s, y_train)
prob_cw = lgbm_cw.predict_proba(X_test_s)[:, 1]
costs_cw = threshold_sweep(y_test.values, prob_cw, THRESHOLDS)
t_opt_cw = float(THRESHOLDS[int(np.argmin(costs_cw))])
print(f"Empirical optimum threshold (cw)   : {t_opt_cw:.4f}")

# ─── 4C. Variant 3 — SMOTE-rebalanced training (replicates notebook) ─────────
print("\n" + "─" * 70)
print("Variant 3: LightGBM on SMOTE-rebalanced data (replicates notebook)")
print("─" * 70)
X_tr_smote, y_tr_smote = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
print(f"Post-SMOTE train shape: {X_tr_smote.shape}, positive rate: {y_tr_smote.mean():.4f}")
lgbm_smote = LGBMClassifier(**BASE_PARAMS)
lgbm_smote.fit(X_tr_smote, y_tr_smote)
prob_smote = lgbm_smote.predict_proba(X_test_s)[:, 1]
costs_sm = threshold_sweep(y_test.values, prob_smote, THRESHOLDS)
t_opt_sm = float(THRESHOLDS[int(np.argmin(costs_sm))])
print(f"Empirical optimum threshold (SMOTE): {t_opt_sm:.4f}")

# ─── 4D. Variant 4 — Raw data + Platt / isotonic calibration ─────────────────
print("\n" + "─" * 70)
print("Variant 4: Raw LightGBM + isotonic calibration (CalibratedClassifierCV)")
print("─" * 70)
lgbm_for_cal = LGBMClassifier(**BASE_PARAMS)
cal = CalibratedClassifierCV(lgbm_for_cal, method="isotonic", cv=5)
cal.fit(X_train_s, y_train)
prob_cal = cal.predict_proba(X_test_s)[:, 1]
costs_cal = threshold_sweep(y_test.values, prob_cal, THRESHOLDS)
t_opt_cal = float(THRESHOLDS[int(np.argmin(costs_cal))])
print(f"Empirical optimum threshold (cal)  : {t_opt_cal:.4f}")


# ─── 5. Report ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARISON — cost-optimal thresholds and metrics")
print("=" * 70)

rows = [
    evaluate_at(0.50,    y_test.values, prob_raw,   "Raw       @ t=0.50"),
    evaluate_at(t_opt_raw,y_test.values, prob_raw,   "Raw       @ t=t_opt"),
    evaluate_at(0.50,    y_test.values, prob_cw,    "ClassWt   @ t=0.50"),
    evaluate_at(t_opt_cw, y_test.values, prob_cw,    "ClassWt   @ t=t_opt"),
    evaluate_at(0.50,    y_test.values, prob_smote, "SMOTE     @ t=0.50"),
    evaluate_at(t_opt_sm, y_test.values, prob_smote, "SMOTE     @ t=t_opt"),
    evaluate_at(0.50,    y_test.values, prob_cal,   "Calibrated@ t=0.50"),
    evaluate_at(t_opt_cal,y_test.values, prob_cal,   "Calibrated@ t=t_opt"),
]
results = pd.DataFrame(rows)
print(results.to_string(index=False))
results.to_csv(OUT_CSV, index=False)
print(f"\nResults saved to: {OUT_CSV}")

# ─── 6. Interpretation ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
t_star = COST_FP / (COST_FP + COST_FN)
print(f"Theoretical t* = CFP/(CFN+CFP) = {t_star:.4f}")
print(f"{'Variant':<22} {'t_opt':>8} {'|gap|':>8}")
for label, t in [("Raw (no SMOTE/CW)", t_opt_raw),
                 ("Class-weighted",    t_opt_cw),
                 ("SMOTE",              t_opt_sm),
                 ("Calibrated (isotonic)", t_opt_cal)]:
    print(f"{label:<22} {t:>8.4f} {abs(t-t_star):>8.4f}")

print("\nReading the table:")
print("  • If the Calibrated variant's t_opt ≈ 0.12 → the asymmetry is a")
print("    CALIBRATION artefact (fixed by post-hoc calibration).")
print("  • If Raw (unweighted) is also ≈ 0.12 → SMOTE/class-weighting was the")
print("    cause of the miscalibration in your notebook.")
print("  • If Raw still gives a very low t_opt (<0.05) → the gap is caused by")
print("    LightGBM's raw margin scores, not by imbalance handling, and the")
print("    theoretical t* simply doesn't apply to uncalibrated boosted trees.")
