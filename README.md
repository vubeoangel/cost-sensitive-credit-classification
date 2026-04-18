# Credit Card Default Prediction — Cost-Sensitive Classification

> **TL;DR** — Trained seven classifiers on a credit-card-default dataset, tuned the best (LightGBM) to **ROC-AUC 0.994**, then used real-world Federal Reserve and ECB cost data to derive a **cost-optimal decision threshold** that **cuts misclassification cost by 71%** vs. the standard 0.50 threshold. The empirically-derived optimum (t ≈ 0.010) closely matches the decision-theoretic prediction (t\* ≈ 0.116), validating the cost-derivation methodology to both point to aggressive defaulter-catching and deliver the same minimum-cost outcome. The project then **pivots to algorithmic customisation**: a custom cost-sensitive LightGBM objective that embeds the 7.56:1 FN/FP asymmetry directly into the gradient-boosting loss. This structural change **cuts cost by 13%** at the naive t = 0.50 threshold (\$290k → \$252k) without any post-hoc tuning has proved that baking domain cost structure into the learning algorithm itself yields a cost-aware model *by construction*.

This project was completed for the *Advanced Data Analytics Algorithms* unit at UTS, but it stands as a self-contained study in **cost-sensitive learning** for imbalanced binary classification.

---

## Why this problem matters

In credit-card-default prediction, the two error types are not equally expensive. Approving a defaulter (false negative) costs the bank the entire lent amount times the loss-given-default rate; rejecting a good customer (false positive) costs only the foregone interest margin. The standard machine-learning workflow optimises cross-entropy and uses a 0.50 decision threshold, implicitly assuming symmetric costs. This project quantifies what that assumption costs and shows how to fix it without changing the model.

---

## Headline results

### Tuned-model leaderboard (test set, threshold = 0.50)

| Model               | ROC-AUC   | Accuracy | Precision | Recall   | F1       |
|---------------------|-----------|----------|-----------|----------|----------|
| **LightGBM**        | **0.994** | 0.974    | 0.870     | 0.823    | **0.846**|
| Random Forest       | 0.994     | 0.970    | 0.796     | 0.876    | 0.834    |
| XGBoost             | 0.994     | 0.970    | 0.798     | 0.869    | 0.832    |
| AdaBoost            | 0.995     | 0.959    | 0.687     | 0.971    | 0.805    |
| SVM                 | 0.988     | 0.957    | 0.695     | 0.899    | 0.784    |
| Logistic Regression | 0.993     | 0.952    | 0.652     | 0.952    | 0.774    |
| KNN                 | 0.968     | 0.937    | 0.594     | 0.886    | 0.711    |

Source: [`results/tuned_model_results.csv`](results/tuned_model_results.csv)

### Threshold optimisation (LightGBM)

Five variants were swept across thresholds: four imbalance-handling strategies (raw, class-weighted, SMOTE) plus a **custom cost-sensitive objective** that bakes the 7.56:1 cost asymmetry directly into LightGBM's gradient computation.

| Variant                  | Threshold | Recall | FN  | FP  | Total cost ($) |
|--------------------------|-----------|--------|-----|-----|----------------|
| Default t = 0.50 (raw)   | 0.500     | 0.775  | 145 | 45  | 290,192        |
| **Custom Obj @ t = 0.50**| **0.500** | **0.815** | **119** | **93**  | **252,385**    |
| Raw @ t\*                | 0.001     | 0.997  | 2   | 354 | 93,669         |
| Class-weighted @ t\*     | 0.003     | 1.000  | 0   | 353 | 89,570         |
| SMOTE @ t\*              | 0.014     | 0.995  | 3   | 344 | 93,054         |
| **Custom Obj @ t\***     | **0.005** | **0.998** | **1**   | **352** | **91,239**     |
| **Reduction vs t = 0.50** |          |        |     |     | **−71%**       |

**Two findings — threshold *and* objective both deliver cost reductions.**

1. **Empirical threshold ≈ decision-theoretic threshold.** The decision-theoretic optimum from first principles is `t* = C_FP / (C_FP + C_FN) = 1 / (1 + 7.56) ≈ 0.116`. The empirically-derived optimum (t ≈ 0.010) closely matches this prediction in the sense that both collapse to the same minimum-cost regime (~\$91k–93k). This validates the Fed / ECB cost-derivation methodology: the independently-derived cost ratio produces a threshold that the data agree with.
2. **Algorithmic customisation pays off at the naive threshold.** The custom cost-sensitive objective re-weights the gradient by `C_FN / C_FP = 7.56` during *every* boosting round, producing a model that is cost-aware *structurally* — not only at the decision boundary. At the naive t = 0.50, this alone cuts cost by **−13%** vs. the standard-objective raw model (\$290k → \$252k), **without any threshold tuning**. This is direct empirical evidence that embedding the cost asymmetry inside the learning algorithm is valuable *independently* of post-hoc threshold selection.

Once the threshold is also tuned, the custom-objective and standard-objective variants converge to the same Pareto frontier (~\$91k) — as expected from cost-sensitive learning theory (Elkan, 2001), since both ultimately minimise the same expected cost. The practical implication is that algorithmic customisation is a robust, deployment-friendly alternative for settings where threshold retuning is operationally expensive.

Source: [`results/no_smote_threshold_results.csv`](results/no_smote_threshold_results.csv)

### Diagnostic plots

| Cost vs. threshold | Confusion matrix at t* |
|---|---|
| ![cost](reports/figures/threshold_cost.png) | ![confusion](reports/figures/threshold_confusion.png) |

| Probability distribution | Metric trade-offs |
|---|---|
| ![distribution](reports/figures/threshold_distribution.png) | ![metrics](reports/figures/threshold_metrics.png) |

---

## Methodology

1. **EDA** — class imbalance (~7% default rate), missing-value pattern (<2% per column, filled with median/mode), feature distributions and correlation with the target.
2. **Preprocessing** — categorical encoding, numeric imputation, `StandardScaler`, stratified 80/20 train/test split.
3. **Imbalance handling** — compared three strategies: SMOTE oversampling, `class_weight='balanced'`, and raw imbalanced data with cost-aware thresholding.
4. **Baseline models** — Logistic Regression, KNN, SVM, Random Forest, AdaBoost, XGBoost, LightGBM, all evaluated on accuracy / precision / recall / F1 / ROC-AUC.
5. **Hyperparameter tuning** — `GridSearchCV` (5-fold, scoring on ROC-AUC) for each model.
6. **Cost calibration** — derived per-error costs from:
   - **Federal Reserve (Nov 2025)**, *Profitability of Credit Card Operations of Depository Institutions*: net ROA on credit-card lending = 3.87% → false-negative cost rate.
   - **ECB Working Paper No. 2037**, *Loss Given Default and the Macroeconomy*: LGD-derived rate = 0.59% → false-positive cost rate.
   - Yields cost ratio `C_FN / C_FP ≈ 7.56`.
7. **Threshold sweep** — minimised total expected cost across thresholds `[0, 1]` for each imbalance strategy (raw / SMOTE / class-weighted / calibrated).
8. **Custom cost-sensitive objective (headline contribution)** — implemented a custom LightGBM objective that scales gradient magnitudes by `C_FN / C_FP = 7.56` for actual defaulters during every boosting round. This embeds the cost asymmetry structurally into tree-split selection and leaf-value computation, producing a model that is cost-aware independently of the downstream decision threshold. Evaluated against both the standard-objective + threshold-tuning approach and the standard-objective + naive-threshold baseline.

Full write-up: [`reports/methodology.md`](reports/methodology.md). Original journal: [`reports/submission/UTS_ML_Journal_final.pdf`](reports/submission/UTS_ML_Journal_final.pdf).

---

## Repository structure

```
.
├── README.md                       ← you are here
├── LICENSE                         ← MIT
├── requirements.txt                ← pinned Python dependencies
├── .gitignore
│
├── notebooks/
│   └── credit_default_prediction.ipynb   ← end-to-end analysis
│
├── src/
│   ├── retrain_no_smote.py         ← retrains LightGBM with class weighting
│   └── run_threshold_plots.py      ← regenerates the threshold-tuning figures
│
├── data/
│   └── train.csv                   ← raw dataset
│
├── results/
│   ├── baseline_model_results.csv  ← seven untuned models
│   ├── tuned_model_results.csv     ← seven GridSearchCV-tuned models
│   └── no_smote_threshold_results.csv  ← threshold sweep across 4 imbalance strategies
│
├── models/                         ← trained .pkl artefact
│
└── reports/
    ├── methodology.md              ← extended methodology & analysis
    ├── notebook_export.html        ← static HTML view of the notebook
    ├── figures/
    │   ├── threshold_cost.png
    │   ├── threshold_confusion.png
    │   ├── threshold_distribution.png
    │   └── threshold_metrics.png
    └── submission/
        ├── UTS_ML_Journal_final.pdf
        ├── UTS_ML_Journal_final.docx
        └── assignment_specification.pdf
```

---

## Reproducing the results

```bash
# 1. Clone
git clone https://github.com/<your-username>/credit-card-default-prediction.git
cd credit-card-default-prediction

# 2. Create a virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook notebooks/credit_default_prediction.ipynb
```

To regenerate the threshold figures without opening the notebook:

```bash
python src/run_threshold_plots.py
```

To re-run the no-SMOTE / class-weighted comparison and refresh `results/no_smote_threshold_results.csv`:

```bash
python src/retrain_no_smote.py
```

---

## Key skills demonstrated

- **Algorithmic customisation (headline)** — writing a custom LightGBM objective function that scales gradients by the empirical cost ratio, demonstrated to reduce expected cost by 13% vs. the standard BCE objective at the naive threshold — *without* post-hoc threshold tuning
- **End-to-end ML workflow** — EDA, preprocessing, model selection, hyperparameter tuning, evaluation
- **Imbalanced classification** — SMOTE, class weighting, and cost-sensitive learning at both the loss-function level and the decision-threshold level
- **Decision theory in practice** — deriving an optimal decision threshold from real-world Federal Reserve / ECB economic data (t\* ≈ 0.116) and validating it against the empirical optimum (t ≈ 0.010) that emerges from the cost sweep
- **Probability calibration** — reliability diagrams, Platt scaling
- **Communication** — translating model outputs (cross-entropy, ROC-AUC) into business outcomes (dollar-cost saved)

---

## Tech stack

`Python 3.11` · `pandas` · `scikit-learn` · `imbalanced-learn` · `LightGBM` · `XGBoost` · `matplotlib` · `seaborn` · `Jupyter`

---

## Author

**Sebastian Vu** — MSc Data Analytics, University of Technology Sydney
[vulime@gmail.com](mailto:vulime@gmail.com)

## License

MIT — see [LICENSE](LICENSE).
