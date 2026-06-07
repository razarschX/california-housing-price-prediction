"""
California Housing Price Prediction
Mathematical Methods in Data Analysis and Machine Learning
Abdullah Gul University

This script implements:
  1. Data loading and exploration
  2. Data cleaning and feature engineering
  3. Train / validation / test split
  4. Baseline OLS (MLE) via normal equations
  5. Ridge regression (MAP) with cross-validated lambda
  6. Polynomial feature expansion (degree 2) with ridge
  7. Evaluation: RMSE, MAE, R²
  8. Visualisations saved to figures/

Run: python analysis.py
All figures are saved automatically.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# ── reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── output directories ───────────────────────────────────────────────────────
FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# ── plot style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})
PALETTE = {"ols": "#2166ac", "ridge": "#d6604d", "poly": "#4dac26", "base": "#888888"}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(path: str = "data/housing.csv") -> pd.DataFrame:
    """Load the California Housing dataset from the local CSV."""
    df = pd.read_csv(path)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_eda(df: pd.DataFrame) -> None:
    """
    Figure 1 — EDA overview:
      (a) Target distribution (raw and log)
      (b) Correlation heatmap
      (c) Median income vs house value scatter
    """
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # (a) Raw price histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(df["median_house_value"] / 1e5, bins=50, color=PALETTE["ols"], alpha=0.8)
    ax1.set_xlabel("Median house value (×$100 k)")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Price distribution")

    # (b) Log-price histogram
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(np.log1p(df["median_house_value"]), bins=50, color=PALETTE["ridge"], alpha=0.8)
    ax2.set_xlabel("log(1 + price)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Log-price distribution")

    # (c) Income vs price scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(df["median_income"], df["median_house_value"] / 1e5,
                alpha=0.15, s=4, color=PALETTE["ols"])
    ax3.set_xlabel("Median household income ($10 k)")
    ax3.set_ylabel("Median house value (×$100 k)")
    ax3.set_title("(c) Income vs house value")

    # (d) Correlation heatmap (numeric columns only)
    ax4 = fig.add_subplot(gs[1, :])
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, ax=ax4, cmap="RdBu_r", vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                linewidths=0.4, cbar_kws={"shrink": 0.7})
    ax4.set_title("(d) Feature correlation matrix")
    ax4.tick_params(axis="x", rotation=45)

    fig.suptitle("Figure 1 — Exploratory Data Analysis: California Housing",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(FIGURES / "fig1_eda.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig1_eda.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. DATA CLEANING AND FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning steps:
      - Drop rows with missing total_bedrooms (207 rows, ~1%)
      - One-hot encode ocean_proximity
      - Engineer per-household features (rooms_per_household,
        bedrooms_per_room, population_per_household)
      - Log-transform the target (median_house_value) to reduce skew
    """
    df = df.copy()

    # Missing values
    n_missing = df["total_bedrooms"].isna().sum()
    print(f"  Missing values in total_bedrooms: {n_missing} — dropping rows")
    df = df.dropna(subset=["total_bedrooms"]).reset_index(drop=True)

    # Derived features
    df["rooms_per_household"]      = df["total_rooms"]    / df["households"]
    df["bedrooms_per_room"]        = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"]     / df["households"]

    # One-hot encode ocean_proximity (drop first to avoid perfect multicollinearity)
    df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True, dtype=float)

    # Log-transform target
    df["log_price"] = np.log1p(df["median_house_value"])

    print(f"  Dataset after cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_feature_matrix(df: pd.DataFrame):
    """
    Returns X (feature matrix, NumPy) and y (log-price vector, NumPy).
    Drops the raw target and original total_rooms / total_bedrooms / population
    (captured by per-household versions).
    """
    drop_cols = ["median_house_value", "log_price",
                 "total_rooms", "total_bedrooms", "population"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(float)
    y = df["log_price"].values.astype(float)
    feature_names = feature_cols
    return X, y, feature_names


# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / VALIDATION / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def split_data(X: np.ndarray, y: np.ndarray,
               val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Splits data into train / validation / test (70 / 15 / 15) using
    random index shuffling — no sklearn dependency for the split itself.
    """
    n = len(y)
    idx = np.random.permutation(n)

    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)

    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    print(f"  Split — train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
# 5. FEATURE STANDARDISATION
# ═══════════════════════════════════════════════════════════════════════════

def standardise(X_train, X_val, X_test):
    """
    Standardise features using training-set mean and std only
    (prevents data leakage from val/test into training).
    Returns scaled arrays and the scaler parameters.
    """
    mu  = X_train.mean(axis=0)
    sig = X_train.std(axis=0) + 1e-8   # avoid division by zero

    X_train_s = (X_train - mu) / sig
    X_val_s   = (X_val   - mu) / sig
    X_test_s  = (X_test  - mu) / sig

    return X_train_s, X_val_s, X_test_s, mu, sig


# ═══════════════════════════════════════════════════════════════════════════
# 6. MODEL IMPLEMENTATIONS (from scratch, no sklearn for fitting)
# ═══════════════════════════════════════════════════════════════════════════

def augment(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones for the intercept (augmentation trick, Week 8)."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def ols_fit(X_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    OLS normal equation:  θ_MLE = (X^T X)^{-1} X^T y
    Corresponds to MLE under Gaussian noise (Lecture 8).
    """
    return np.linalg.lstsq(X_aug.T @ X_aug, X_aug.T @ y, rcond=None)[0]


def ridge_fit(X_aug: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Ridge (MAP) normal equation:  θ_MAP = (X^T X + λ I)^{-1} X^T y
    The λ I term is NOT applied to the bias weight (index 0), following
    standard convention.  Corresponds to Gaussian prior on θ (Lecture 8).
    """
    n_feat = X_aug.shape[1]
    reg = lam * np.eye(n_feat)
    reg[0, 0] = 0.0                          # do not penalise the intercept
    return np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)


def predict(X_aug: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return X_aug @ theta


# ═══════════════════════════════════════════════════════════════════════════
# 7. EVALUATION METRICS
# ═══════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1 - ss_res / ss_tot)

def evaluate(name, y_true, y_pred):
    """Print and return a metrics dictionary."""
    m = {"model": name,
         "RMSE": rmse(y_true, y_pred),
         "MAE":  mae(y_true, y_pred),
         "R2":   r2(y_true, y_pred)}
    print(f"  {name:<30s}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  R²={m['R2']:.4f}")
    return m


# ═══════════════════════════════════════════════════════════════════════════
# 8. CROSS-VALIDATION FOR LAMBDA SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def cross_validate_lambda(X_train_s, y_train, lambdas, k=5):
    """
    k-fold cross-validation on the training set to choose λ.
    Returns a dict mapping lambda → mean validation RMSE.
    """
    n = len(y_train)
    fold_size = n // k
    cv_rmses = {}

    for lam in lambdas:
        fold_rmses = []
        for fold in range(k):
            val_start = fold * fold_size
            val_end   = val_start + fold_size
            mask = np.zeros(n, dtype=bool)
            mask[val_start:val_end] = True

            Xf_val,   yf_val   = X_train_s[mask],  y_train[mask]
            Xf_train, yf_train = X_train_s[~mask], y_train[~mask]

            Xa_train = augment(Xf_train)
            Xa_val   = augment(Xf_val)
            theta    = ridge_fit(Xa_train, yf_train, lam)
            fold_rmses.append(rmse(yf_val, predict(Xa_val, theta)))

        cv_rmses[lam] = float(np.mean(fold_rmses))

    return cv_rmses


# ═══════════════════════════════════════════════════════════════════════════
# 9. POLYNOMIAL FEATURES (degree 2)
# ═══════════════════════════════════════════════════════════════════════════

def polynomial_features_degree2(X: np.ndarray) -> np.ndarray:
    """
    Expand features to degree-2 polynomial basis:
      original features + pairwise products + squares.
    Only uses the first 8 features (to keep dimensionality manageable).
    """
    X_sub = X[:, :8]                        # use first 8 numeric features
    n, d  = X_sub.shape
    cols  = [X_sub]                         # degree-1

    # degree-2: x_i * x_j for i <= j
    for i in range(d):
        for j in range(i, d):
            cols.append((X_sub[:, i] * X_sub[:, j]).reshape(-1, 1))

    return np.hstack(cols)


# ═══════════════════════════════════════════════════════════════════════════
# 10. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

def plot_cv_lambda(lambdas, cv_rmses, best_lam):
    """Figure 2 — Cross-validation RMSE vs λ (regularisation selection)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    lam_arr = np.array(lambdas)
    rmse_arr = np.array([cv_rmses[l] for l in lambdas])

    ax.semilogx(lam_arr, rmse_arr, "o-", color=PALETTE["ridge"], lw=2, ms=5)
    ax.axvline(best_lam, color=PALETTE["ols"], ls="--", lw=1.5,
               label=f"Best λ = {best_lam:.4f}")
    ax.set_xlabel("Regularisation parameter λ (log scale)")
    ax.set_ylabel("5-fold CV RMSE (log-price)")
    ax.set_title("Figure 2 — Cross-validation RMSE vs λ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_cv_lambda.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig2_cv_lambda.png")


def plot_regularisation_path(X_aug, y, lambdas, feature_names):
    """Figure 3 — Regularisation path: how coefficients shrink with λ."""
    coef_matrix = []
    for lam in lambdas:
        theta = ridge_fit(X_aug, y, lam)
        coef_matrix.append(theta[1:])          # exclude intercept

    coef_matrix = np.array(coef_matrix)        # (n_lambdas, n_features)

    # Plot only the 8 features with largest range across λ
    ranges = coef_matrix.max(axis=0) - coef_matrix.min(axis=0)
    top_idx = np.argsort(ranges)[::-1][:8]

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in top_idx:
        ax.semilogx(lambdas, coef_matrix[:, i],
                    label=feature_names[i] if i < len(feature_names) else f"feat {i}",
                    lw=1.8)

    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xlabel("λ (log scale)")
    ax.set_ylabel("Coefficient value")
    ax.set_title("Figure 3 — Ridge regularisation path (top 8 features by range)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_regularisation_path.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig3_regularisation_path.png")


def plot_predicted_vs_actual(results: dict, y_test: np.ndarray):
    """Figure 4 — Predicted vs actual for OLS, Ridge, and Polynomial+Ridge."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    models = [("OLS (MLE)",    "ols",  PALETTE["ols"]),
              ("Ridge (MAP)",  "ridge",PALETTE["ridge"]),
              ("Poly+Ridge",   "poly", PALETTE["poly"])]

    for ax, (name, key, col) in zip(axes, models):
        y_pred = results[key]
        ax.scatter(y_test, y_pred, alpha=0.25, s=6, color=col)
        mn = min(y_test.min(), y_pred.min())
        mx = max(y_test.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1, label="Perfect fit")
        ax.set_xlabel("Actual log-price")
        ax.set_ylabel("Predicted log-price")
        ax.set_title(name)
        r2_val = r2(y_test, y_pred)
        ax.text(0.05, 0.93, f"R² = {r2_val:.3f}", transform=ax.transAxes,
                fontsize=9, color=col)

    fig.suptitle("Figure 4 — Predicted vs Actual (log-price), test set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_predicted_vs_actual.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig4_predicted_vs_actual.png")


def plot_residuals(results: dict, y_test: np.ndarray):
    """Figure 5 — Residual plots for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    models = [("OLS (MLE)",   "ols",  PALETTE["ols"]),
              ("Ridge (MAP)", "ridge",PALETTE["ridge"]),
              ("Poly+Ridge",  "poly", PALETTE["poly"])]

    for ax, (name, key, col) in zip(axes, models):
        y_pred = results[key]
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.25, s=6, color=col)
        ax.axhline(0, color="k", lw=1, ls="--")
        ax.set_xlabel("Fitted value")
        ax.set_ylabel("Residual (actual − predicted)")
        ax.set_title(name)
        rmse_val = rmse(y_test, y_pred)
        ax.text(0.05, 0.93, f"RMSE = {rmse_val:.4f}", transform=ax.transAxes,
                fontsize=9, color=col)

    fig.suptitle("Figure 5 — Residual plots (test set)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig5_residuals.png")


def plot_overfitting_curve(X_train_s, y_train, X_val_s, y_val):
    """
    Figure 6 — Overfitting demonstration: training vs validation RMSE
    for OLS as the number of training samples grows (learning curve).
    """
    sizes = np.linspace(50, len(y_train), 25, dtype=int)
    train_rmses, val_rmses = [], []

    for n in sizes:
        Xs = augment(X_train_s[:n])
        theta = ols_fit(Xs, y_train[:n])
        train_rmses.append(rmse(y_train[:n], predict(Xs, theta)))
        val_rmses.append(rmse(y_val, predict(augment(X_val_s), theta)))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, train_rmses, "o-", color=PALETTE["ols"],   lw=2, ms=4, label="Training RMSE")
    ax.plot(sizes, val_rmses,   "s-", color=PALETTE["ridge"], lw=2, ms=4, label="Validation RMSE")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("RMSE (log-price)")
    ax.set_title("Figure 6 — Learning curve (OLS): training vs validation RMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_learning_curve.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig6_learning_curve.png")


def plot_model_comparison(metrics_list):
    """Figure 7 — Bar chart comparing RMSE, MAE, R² across all models."""
    df_m = pd.DataFrame(metrics_list)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colours = [PALETTE["base"], PALETTE["ols"], PALETTE["ridge"], PALETTE["poly"]]
    metric_labels = {"RMSE": "RMSE (log-price)", "MAE": "MAE (log-price)", "R2": "R²"}

    for ax, (col, label) in zip(axes, metric_labels.items()):
        bars = ax.bar(df_m["model"], df_m[col], color=colours, edgecolor="white", width=0.55)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{bar.get_height():.4f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle("Figure 7 — Model comparison on the test set",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig7_model_comparison.png")


def plot_coefficient_importance(theta, feature_names):
    """Figure 8 — Top-15 ridge coefficients by absolute magnitude."""
    coefs = theta[1:]                           # exclude intercept
    n_show = min(15, len(coefs))
    idx    = np.argsort(np.abs(coefs))[::-1][:n_show]
    names  = [feature_names[i] if i < len(feature_names) else f"feat {i}" for i in idx]
    vals   = coefs[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [PALETTE["ridge"] if v > 0 else PALETTE["ols"] for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], edgecolor="white")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Coefficient (standardised features)")
    ax.set_title("Figure 8 — Ridge regression: top 15 coefficients")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig8_coefficients.png", bbox_inches="tight")
    plt.close(fig)
    print("  [saved] fig8_coefficients.png")


# ═══════════════════════════════════════════════════════════════════════════
# 11. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 60)
    print("  CALIFORNIA HOUSING PRICE PREDICTION")
    print("  Mathematical Methods in Data Analysis & ML")
    print("═" * 60 + "\n")

    # ── Step 1: Load ──────────────────────────────────────────────────────
    print("▶ 1. Loading data...")
    df_raw = load_data()

    # ── Step 2: EDA ───────────────────────────────────────────────────────
    print("\n▶ 2. Exploratory data analysis...")
    print(df_raw.describe().round(2).to_string())
    print(f"\n  Missing values:\n{df_raw.isna().sum()[df_raw.isna().sum() > 0]}")
    plot_eda(df_raw)

    # ── Step 3: Clean ────────────────────────────────────────────────────
    print("\n▶ 3. Cleaning and feature engineering...")
    df = clean_and_engineer(df_raw)
    X, y, feature_names = build_feature_matrix(df)
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Features: {feature_names}")

    # ── Step 4: Split ────────────────────────────────────────────────────
    print("\n▶ 4. Train / validation / test split (70/15/15)...")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # ── Step 5: Standardise ──────────────────────────────────────────────
    print("\n▶ 5. Standardising features (train statistics only)...")
    X_train_s, X_val_s, X_test_s, mu, sig = standardise(X_train, X_val, X_test)

    # Augmented matrices (intercept column prepended)
    Xa_train = augment(X_train_s)
    Xa_val   = augment(X_val_s)
    Xa_test  = augment(X_test_s)

    # ── Step 6: Baseline (mean predictor) ─────────────────────────────────
    print("\n▶ 6. Baseline model (mean predictor)...")
    y_pred_base = np.full_like(y_test, y_train.mean())
    m_base = evaluate("Baseline (mean)", y_test, y_pred_base)

    # ── Step 7: OLS (MLE) ─────────────────────────────────────────────────
    print("\n▶ 7. OLS (MLE) via normal equations...")
    theta_ols = ols_fit(Xa_train, y_train)
    y_pred_ols_val  = predict(Xa_val,  theta_ols)
    y_pred_ols_test = predict(Xa_test, theta_ols)
    m_ols = evaluate("OLS (MLE)", y_test, y_pred_ols_test)
    print(f"  Validation RMSE: {rmse(y_val, y_pred_ols_val):.4f}")

    # ── Step 8: Ridge (MAP) with cross-validated λ ──────────────────────
    print("\n▶ 8. Ridge (MAP) — 5-fold CV for λ selection...")
    lambdas = np.logspace(-4, 3, 50)
    cv_rmses = cross_validate_lambda(X_train_s, y_train, lambdas, k=5)
    best_lam = min(cv_rmses, key=cv_rmses.get)
    print(f"  Best λ = {best_lam:.4f}  (CV RMSE = {cv_rmses[best_lam]:.4f})")

    theta_ridge = ridge_fit(Xa_train, y_train, best_lam)
    y_pred_ridge_test = predict(Xa_test, theta_ridge)
    m_ridge = evaluate("Ridge (MAP)", y_test, y_pred_ridge_test)

    # ── Step 9: Polynomial degree-2 + Ridge ──────────────────────────────
    print("\n▶ 9. Polynomial features (degree 2) + Ridge...")
    X_train_p = polynomial_features_degree2(X_train_s)
    X_val_p   = polynomial_features_degree2(X_val_s)
    X_test_p  = polynomial_features_degree2(X_test_s)

    # Re-standardise polynomial features
    mu_p  = X_train_p.mean(axis=0)
    sig_p = X_train_p.std(axis=0) + 1e-8
    X_train_ps = (X_train_p - mu_p) / sig_p
    X_val_ps   = (X_val_p   - mu_p) / sig_p
    X_test_ps  = (X_test_p  - mu_p) / sig_p

    # Cross-validate λ for polynomial model
    cv_rmses_poly = cross_validate_lambda(X_train_ps, y_train, lambdas, k=5)
    best_lam_poly = min(cv_rmses_poly, key=cv_rmses_poly.get)
    print(f"  Best λ (poly) = {best_lam_poly:.4f}")

    Xa_train_ps = augment(X_train_ps)
    Xa_test_ps  = augment(X_test_ps)
    theta_poly  = ridge_fit(Xa_train_ps, y_train, best_lam_poly)
    y_pred_poly_test = predict(Xa_test_ps, theta_poly)
    m_poly = evaluate("Polynomial deg-2 + Ridge", y_test, y_pred_poly_test)

    # ── Step 10: All figures ──────────────────────────────────────────────
    print("\n▶ 10. Generating figures...")
    plot_cv_lambda(lambdas, cv_rmses, best_lam)
    plot_regularisation_path(Xa_train, y_train, lambdas, feature_names)
    results = {"ols":   y_pred_ols_test,
               "ridge": y_pred_ridge_test,
               "poly":  y_pred_poly_test}
    plot_predicted_vs_actual(results, y_test)
    plot_residuals(results, y_test)
    plot_overfitting_curve(X_train_s, y_train, X_val_s, y_val)
    metrics_list = [m_base, m_ols, m_ridge, m_poly]
    plot_model_comparison(metrics_list)
    plot_coefficient_importance(theta_ridge, feature_names)

    # ── Step 11: Summary table ────────────────────────────────────────────
    print("\n▶ 11. Final results summary (test set)")
    print("─" * 60)
    print(f"  {'Model':<32s}  {'RMSE':>7s}  {'MAE':>7s}  {'R²':>7s}")
    print("─" * 60)
    for m in metrics_list:
        print(f"  {m['model']:<32s}  {m['RMSE']:>7.4f}  {m['MAE']:>7.4f}  {m['R2']:>7.4f}")
    print("─" * 60)

    # ── Key metrics for report ────────────────────────────────────────────
    print("\n▶ 12. Key values for the project report")
    print(f"  Dataset: {len(df)} samples, {X.shape[1]} features after engineering")
    print(f"  Train/val/test: {len(y_train)}/{len(y_val)}/{len(y_test)}")
    print(f"  Best λ (ridge): {best_lam:.4f}")
    print(f"  Best λ (poly+ridge): {best_lam_poly:.4f}")
    print(f"  OLS test R²:   {m_ols['R2']:.4f}")
    print(f"  Ridge test R²: {m_ridge['R2']:.4f}")
    print(f"  Poly test R²:  {m_poly['R2']:.4f}")
    print(f"  Ridge RMSE improvement over OLS: "
          f"{(m_ols['RMSE'] - m_ridge['RMSE']) / m_ols['RMSE'] * 100:.4f}%")
    print("\n✓ All figures saved to figures/")
    print("✓ Analysis complete.\n")


if __name__ == "__main__":
    main()