"""
Threshold Tuning Plots – LightGBM (Credit Card Default)
Regenerates the threshold-sweep figures used in the README and methodology report.
Outputs are written to ./threshold_outputs/ (relative to this script).

Run from the project root with the project venv activated:
  python src/run_threshold_plots.py
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as mtick
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_auc_score
)
import os

# ── Output directory ────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "threshold_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Palette (from notebook) ──────────────────────────────────────────
PALETTE = ["#7C3AED", "#38BDF8", "#A78BFA", "#60A5FA", "#C084FC"]

# ── Seaborn / rcParams (from notebook) ──────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.size": 12
})

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("Loading data...")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/train.csv'))

# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING  (exactly as in notebook)
# ══════════════════════════════════════════════════════════════════════
cat_cols = ['gender', 'owns_car', 'owns_house', 'occupation_type']
for col in cat_cols:
    df[col] = df[col].astype('category')

# Fix gender XNA → mode
mode_gender = df['gender'].mode()[0]
df['gender'] = df['gender'].apply(lambda x: mode_gender if x == 'XNA' else x)

# Fill owns_car NA → mode
df['owns_car'] = df['owns_car'].fillna(df['owns_car'].mode()[0])

# Fill numerical NAs with median
exclude_cols = ["customer_id", "name", "credit_card_default"]
cat_feats = [col for col in df.columns if df[col].dtype.name == 'category']
num_feats  = [col for col in df.columns
              if df[col].dtype.name != 'category' and col not in exclude_cols]

missing_cols_num = df[num_feats].columns[df[num_feats].isnull().any()]
for col in missing_cols_num:
    df[col] = df[col].fillna(df[col].median())

# Drop invalid employment days (>36500 ≈ 100 yrs)
df = df[df['no_of_days_employed'] <= 36500].copy()

# Drop irrelevant columns
df = df.drop(columns=["customer_id", "name"])

# Map binary features
binary_maps = {
    'gender':     {'M': 1, 'F': 0},
    'owns_car':   {'Y': 1, 'N': 0},
    'owns_house': {'Y': 1, 'N': 0},
}
for col, mapping in binary_maps.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Feature engineering
df["debt_to_income_ratio"]    = df["yearly_debt_payments"] / df["net_yearly_income"].replace(0, np.nan)
df["children_family_ratio"]   = df["no_of_children"] / df["total_family_members"].replace(0, np.nan)
df["income_per_family_member"] = df["net_yearly_income"] / df["total_family_members"].replace(0, np.nan)

# Drop low-importance features (as decided in notebook)
df = df.drop(columns=["credit_limit", "total_family_members", "owns_house"],
             errors='ignore')

# One-hot encode occupation_type
df = pd.get_dummies(df, columns=["occupation_type"], drop_first=True, dtype=int)

# ══════════════════════════════════════════════════════════════════════
# 3. SPLIT + SCALE + SMOTE
# ══════════════════════════════════════════════════════════════════════
target = "credit_card_default"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# ══════════════════════════════════════════════════════════════════════
# 4. TRAIN TUNED LIGHTGBM  (best params from notebook output)
# ══════════════════════════════════════════════════════════════════════
print("Training LightGBM with best params from notebook...")
best_lgbm = LGBMClassifier(
    random_state=42,
    verbose=-1,
    learning_rate=0.1,
    max_depth=-1,
    n_estimators=200,
    num_leaves=50,
    subsample=0.8
)
best_lgbm.fit(X_train_smote, y_train_smote)

y_prob_lgbm = best_lgbm.predict_proba(X_test_scaled)[:, 1]
print(f"ROC-AUC on test set: {roc_auc_score(y_test, y_prob_lgbm):.4f}")

# ══════════════════════════════════════════════════════════════════════
# 5. COST-BASED THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════
AVG_CL_DEFAULT    = 49679.0
AVG_CL_NONDEFAULT = 43007.0
FN_RATE = 0.0387
FP_RATE = 0.0059
COST_FN = FN_RATE * AVG_CL_DEFAULT     # $1,922.58
COST_FP = FP_RATE * AVG_CL_NONDEFAULT  # $253.74

print(f"\nCost per FN: ${COST_FN:,.2f}")
print(f"Cost per FP: ${COST_FP:,.2f}")
print(f"FN/FP ratio: {COST_FN/COST_FP:.2f}×")

thresholds_sweep = np.linspace(0.01, 0.99, 500)
costs_total_arr, costs_fn_arr, costs_fp_arr = [], [], []

for t in thresholds_sweep:
    y_pred_t = (y_prob_lgbm >= t).astype(int)
    fn = int(((y_pred_t == 0) & (y_test == 1)).sum())
    fp = int(((y_pred_t == 1) & (y_test == 0)).sum())
    costs_fn_arr.append(fn * COST_FN)
    costs_fp_arr.append(fp * COST_FP)
    costs_total_arr.append(fn * COST_FN + fp * COST_FP)

costs_total_arr = np.array(costs_total_arr)
costs_fn_arr    = np.array(costs_fn_arr)
costs_fp_arr    = np.array(costs_fp_arr)

opt_idx       = int(np.argmin(costs_total_arr))
opt_threshold = float(thresholds_sweep[opt_idx])
opt_cost      = float(costs_total_arr[opt_idx])

default_idx  = int(np.argmin(np.abs(thresholds_sweep - 0.5)))
default_cost = float(costs_total_arr[default_idx])

print(f"\nOptimal threshold  : {opt_threshold:.4f}")
print(f"Min total cost     : ${opt_cost:,.2f}")
print(f"Default (0.5) cost : ${default_cost:,.2f}")
print(f"Cost saving        : ${default_cost - opt_cost:,.2f}  "
      f"({(default_cost - opt_cost)/default_cost*100:.1f}% reduction)")

# ══════════════════════════════════════════════════════════════════════
# 6. PLOT 1 – Cost vs Threshold
# ══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.plot(thresholds_sweep, costs_total_arr / 1e3,
        color=PALETTE[0], lw=2.5, label='Total Cost')
ax.plot(thresholds_sweep, costs_fn_arr / 1e3,
        color='#e74c3c', lw=1.8, linestyle='--', label='FN Cost (missed defaulters)')
ax.plot(thresholds_sweep, costs_fp_arr / 1e3,
        color='#3498db', lw=1.8, linestyle='--', label='FP Cost (rejected customers)')
ax.axvline(opt_threshold, color='black', lw=1.5, linestyle=':',
           label=f'Optimal = {opt_threshold:.3f}')
ax.axvline(0.5, color='grey', lw=1.2, linestyle=':', alpha=0.8,
           label='Default = 0.50')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel("Cost ($'000)")
ax.set_title('Cost Decomposition vs Threshold — LightGBM', pad=12)
ax.legend(frameon=False, fontsize=9)
sns.despine(ax=ax)

half_win = 0.20
lo = max(0.01, opt_threshold - half_win)
hi = min(0.99, opt_threshold + half_win)
zoom = (thresholds_sweep >= lo) & (thresholds_sweep <= hi)

ax2 = axes[1]
ax2.plot(thresholds_sweep[zoom], costs_total_arr[zoom] / 1e3,
         color=PALETTE[0], lw=2.5, label='Total Cost')
ax2.axvline(opt_threshold, color='black', lw=1.5, linestyle=':',
            label=f'Optimal = {opt_threshold:.3f}')
ax2.axvline(0.5, color='grey', lw=1.2, linestyle=':', alpha=0.8,
            label='Default = 0.50')
ax2.scatter([opt_threshold], [opt_cost / 1e3],
            color='black', zorder=5, s=90, label=f'Min cost ${opt_cost/1e3:.1f}k')
ax2.set_xlabel('Classification Threshold')
ax2.set_ylabel("Total Cost ($'000)")
ax2.set_title(f'Zoomed View — Optimal Threshold = {opt_threshold:.3f}', pad=12)
ax2.legend(frameon=False, fontsize=9)
sns.despine(ax=ax2)

plt.suptitle('Cost-Based Threshold Optimisation (LightGBM)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, 'plot1_cost_vs_threshold.png')
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out1}")

# ══════════════════════════════════════════════════════════════════════
# 7. PLOT 2 – Precision / Recall / F1 vs Threshold
# ══════════════════════════════════════════════════════════════════════
pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_test, y_prob_lgbm)
f1_curve = 2 * pr_prec * pr_rec / (pr_prec + pr_rec + 1e-8)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(pr_thresh, pr_prec[:-1], color='#2ecc71', lw=2,  label='Precision')
ax.plot(pr_thresh, pr_rec[:-1],  color='#e74c3c', lw=2,  label='Recall')
ax.plot(pr_thresh, f1_curve[:-1], color=PALETTE[0], lw=2, label='F1 Score')
ax.axvline(opt_threshold, color='black', lw=1.5, linestyle=':',
           label=f'Optimal threshold = {opt_threshold:.3f}')
ax.axvline(0.5, color='grey', lw=1.2, linestyle=':', alpha=0.8,
           label='Default threshold = 0.50')
ax.set_xlabel('Classification Threshold')
ax.set_ylabel('Score')
ax.set_title('Precision / Recall / F1 vs Threshold — LightGBM', pad=12)
ax.legend(frameon=False)
ax.set_ylim(0, 1.05)
sns.despine()
plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'plot2_precision_recall_f1.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

# ══════════════════════════════════════════════════════════════════════
# 8. PLOT 3 – Confusion Matrices (default vs optimal threshold)
# ══════════════════════════════════════════════════════════════════════
y_pred_default_t = (y_prob_lgbm >= 0.50).astype(int)
y_pred_optimal_t = (y_prob_lgbm >= opt_threshold).astype(int)

fn_def  = int(((y_pred_default_t == 0) & (y_test == 1)).sum())
fp_def  = int(((y_pred_default_t == 1) & (y_test == 0)).sum())
cost_def = fn_def * COST_FN + fp_def * COST_FP

fn_opt  = int(((y_pred_optimal_t == 0) & (y_test == 1)).sum())
fp_opt  = int(((y_pred_optimal_t == 1) & (y_test == 0)).sum())
cost_opt = fn_opt * COST_FN + fp_opt * COST_FP

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, y_pred, title in zip(
    axes,
    [y_pred_default_t, y_pred_optimal_t],
    [f'Default Threshold (0.50)\nTotal Cost: ${cost_def:,.0f}',
     f'Optimal Threshold ({opt_threshold:.3f})\nTotal Cost: ${cost_opt:,.0f}']
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Default", "Default"]
    )
    disp.plot(cmap="BuPu", values_format="d", ax=ax, colorbar=False)
    for text in disp.text_.ravel():
        text.set_fontsize(12)
        text.set_fontweight("bold")
    ax.set_title(title, pad=14)
    ax.grid(False)

sns.despine(left=True, bottom=True)
plt.suptitle('Confusion Matrix Comparison — LightGBM',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'plot3_confusion_matrices.png')
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out3}")

# ══════════════════════════════════════════════════════════════════════
# 9. COMBINED FIGURE (all 3 plots stacked)
# ══════════════════════════════════════════════════════════════════════
from PIL import Image
imgs = [Image.open(p) for p in [out1, out2, out3]]
widths  = [img.width for img in imgs]
heights = [img.height for img in imgs]
total_h = sum(heights)
max_w   = max(widths)

combined = Image.new('RGB', (max_w, total_h), (255, 255, 255))
y_offset = 0
for img in imgs:
    x_offset = (max_w - img.width) // 2
    combined.paste(img, (x_offset, y_offset))
    y_offset += img.height

out_combined = os.path.join(OUT_DIR, 'threshold_plots_combined.png')
combined.save(out_combined)
print(f"Saved combined: {out_combined}")

print("\nAll plots saved successfully.")
print(f"Output directory: {OUT_DIR}")
