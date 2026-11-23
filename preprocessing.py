"""
Computational Machine Learning - Final Project
Data preprocessing and feature treatment for hypertension prediction (NHANES).

Design principle (leakage-safe):
  1. Row-wise, deterministic transformations (logs, ratios, flags, missingness
     indicators) are applied to the whole dataset up front. They use no
     statistics learned from the data, so they cannot leak information.
  2. Every step that LEARNS something from the data (winsorizing quantiles,
     imputation medians, scaling, splines) lives inside an sklearn Pipeline,
     so it is re-fitted on the training part of each CV fold only.

Usage
  As a script (writes processed files + runs sanity checks):
      python preprocessing.py                 # data in the same folder as this file
      python preprocessing.py --data-dir "C:/path/to/Final Project"
  As a module (from a notebook or the model-search script):
      from preprocessing import load_data, add_features, get_feature_list, \
                                build_preprocessor, make_lockbox_split, get_cv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TARGET = "label"
ID_COL = "id"  # NHANES SEQN: identifier, deliberately NOT used as a feature

CONTINUOUS = ["RIDAGEYR", "BMXBMI", "LBXTC", "LBDHDD", "LBXTR",
              "LBXGLU", "LBXSCR", "LBXSBU"]
BINARY = ["smoker", "vigorous_pa", "moderate_pa"]
RAW_FEATURES = CONTINUOUS + BINARY

# Right-skewed lab values that benefit from a log transform
SKEWED = ["LBXTR", "LBXGLU", "LBXSCR", "LBXSBU"]

# Feature blocks used in the block-wise best-subset search.
# B0 is always included; B1-B5 are optional (2^5 = 32 combinations).
FEATURE_BLOCKS: dict[str, list[str]] = {
    "B0_raw": RAW_FEATURES,
    "B1_logs": [f"log_{c}" for c in SKEWED],
    "B2_ratios": ["tc_hdl_ratio", "log_tg_hdl_ratio", "non_hdl", "bun_cr_ratio"],
    "B3_missing": ["miss_bmi", "miss_lipids", "miss_fasting", "miss_kidney",
                   "n_missing"],
    "B4_interactions": ["age_sq", "age_x_bmi", "age_x_scr", "bmi_x_glu"],
    "B5_flags": ["bmi_overweight", "bmi_obese", "bmi_severe_obese",
                 "glu_prediabetes", "glu_diabetes", "age_topcoded",
                 "any_pa", "pa_score"],
}
OPTIONAL_BLOCKS = [b for b in FEATURE_BLOCKS if b != "B0_raw"]

# Engineered features that are binary/count-like (never winsorized or splined)
DISCRETE_ENGINEERED = (FEATURE_BLOCKS["B3_missing"]
                       + FEATURE_BLOCKS["B5_flags"])


# ---------------------------------------------------------------------------
# 1. Loading and validation
# ---------------------------------------------------------------------------
def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test CSVs and run basic integrity checks."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "hypertension_train.csv")
    test = pd.read_csv(data_dir / "hypertension_test_features.csv")
    validate(train, test)
    train[ID_COL] = train[ID_COL].astype("int64")
    test[ID_COL] = test[ID_COL].astype("int64")
    return train, test


def validate(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Fail loudly if the data is not what the project description says."""
    missing_cols = set(RAW_FEATURES + [ID_COL, TARGET]) - set(train.columns)
    assert not missing_cols, f"Train is missing columns: {missing_cols}"
    missing_cols = set(RAW_FEATURES + [ID_COL]) - set(test.columns)
    assert not missing_cols, f"Test is missing columns: {missing_cols}"

    assert train[ID_COL].is_unique and test[ID_COL].is_unique, "Duplicate ids"
    assert not set(train[ID_COL]) & set(test[ID_COL]), "ids overlap train/test"
    assert set(train[TARGET].unique()) <= {0, 1}, "Label is not binary"
    for c in BINARY:
        for df in (train, test):
            assert set(df[c].dropna().unique()) <= {0, 1}, f"{c} not binary"
    for c in CONTINUOUS:
        for df in (train, test):
            assert (df[c].dropna() > 0).all(), f"{c} has non-positive values"


def describe_data(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Summary table used for the descriptive-analysis slide."""
    rows = []
    for c in RAW_FEATURES:
        rows.append({
            "feature": c,
            "train_missing_%": 100 * train[c].isna().mean(),
            "test_missing_%": 100 * test[c].isna().mean(),
            "train_median": train[c].median(),
            "test_median": test[c].median(),
            "train_p99": train[c].quantile(0.99),
            "max": max(train[c].max(), test[c].max()),
            "rate_if_missing": train.loc[train[c].isna(), TARGET].mean(),
            "rate_if_observed": train.loc[train[c].notna(), TARGET].mean(),
        })
    return pd.DataFrame(rows).set_index("feature").round(3)


# ---------------------------------------------------------------------------
# 2. Deterministic, row-wise feature engineering (no fitted parameters)
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all engineered features added.

    NaNs propagate naturally (e.g. a ratio is NaN if either input is NaN);
    they are handled later, inside the model pipeline.
    """
    out = df.copy()
    age, bmi = out["RIDAGEYR"], out["BMXBMI"]
    tc, hdl, tg = out["LBXTC"], out["LBDHDD"], out["LBXTR"]
    glu, scr, bun = out["LBXGLU"], out["LBXSCR"], out["LBXSBU"]

    # B1 - logs of skewed lab values (all values are strictly positive)
    for c in SKEWED:
        out[f"log_{c}"] = np.log(out[c])

    # B2 - clinically used lipid / kidney ratios
    out["tc_hdl_ratio"] = tc / hdl                 # atherogenic index
    out["log_tg_hdl_ratio"] = np.log(tg / hdl)     # insulin-resistance proxy
    out["non_hdl"] = tc - hdl                      # non-HDL cholesterol
    out["bun_cr_ratio"] = bun / scr                # kidney / hydration status

    # B3 - missingness indicators. Lab blocks are missing together in NHANES:
    # lipids (TC, HDL), fasting subsample (TG, glucose), biochemistry (Cr, BUN).
    out["miss_bmi"] = bmi.isna().astype(int)
    out["miss_lipids"] = (tc.isna() | hdl.isna()).astype(int)
    out["miss_fasting"] = (tg.isna() | glu.isna()).astype(int)
    out["miss_kidney"] = (scr.isna() | bun.isna()).astype(int)
    out["n_missing"] = out[CONTINUOUS].isna().sum(axis=1)

    # B4 - interactions (mainly useful for linear models)
    out["age_sq"] = age ** 2
    out["age_x_bmi"] = age * bmi
    out["age_x_scr"] = age * scr
    out["bmi_x_glu"] = bmi * glu

    # B5 - clinical threshold flags (NaN input -> flag 0, missingness is in B3)
    out["bmi_overweight"] = (bmi >= 25).astype(int)
    out["bmi_obese"] = (bmi >= 30).astype(int)
    out["bmi_severe_obese"] = (bmi >= 40).astype(int)
    out["glu_prediabetes"] = ((glu >= 100) & (glu < 126)).astype(int)
    out["glu_diabetes"] = (glu >= 126).astype(int)
    out["age_topcoded"] = (age >= 80).astype(int)   # NHANES top-codes age at 80
    out["any_pa"] = ((out["vigorous_pa"] == 1)
                     | (out["moderate_pa"] == 1)).astype(int)
    out["pa_score"] = out["vigorous_pa"] * 2 + out["moderate_pa"]

    return out


def get_feature_list(blocks: list[str] | tuple[str, ...] | None = None) -> list[str]:
    """Columns for a given set of optional blocks (B0 is always included).

    blocks=None -> all blocks.  blocks=[] -> raw features only.
    """
    if blocks is None:
        blocks = OPTIONAL_BLOCKS
    unknown = set(blocks) - set(OPTIONAL_BLOCKS)
    assert not unknown, f"Unknown blocks: {unknown}"
    cols = list(FEATURE_BLOCKS["B0_raw"])
    for b in OPTIONAL_BLOCKS:          # keep a stable column order
        if b in blocks:
            cols += FEATURE_BLOCKS[b]
    return cols


# ---------------------------------------------------------------------------
# 3. Fitted transformers (used inside pipelines -> re-fitted per CV fold)
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] quantiles learned on the training
    fold. NaNs are ignored when fitting and kept as NaN when transforming.
    Used for distance/linear models; trees are insensitive to outliers."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor(features: list[str], kind: str = "linear",
                       splines: bool = False, n_knots: int = 5,
                       winsorize: bool = True) -> ColumnTransformer:
    """Return an (unfitted) ColumnTransformer for the given model family.

    kind:
      "linear"      logistic, LDA, SVM, kNN, MLP: winsorize -> median impute
                    -> (optional splines on age & BMI) -> standardize.
      "tree_native" HistGradientBoosting, LightGBM, XGBoost, CatBoost:
                    passthrough; these handle NaN natively and are invariant
                    to monotone transforms and outliers.
      "tree_impute" RandomForest, ExtraTrees (no native NaN in all versions):
                    median imputation only (missingness flags carry the info
                    if block B3 is included).
    """
    cont = [c for c in features if c not in BINARY + DISCRETE_ENGINEERED]
    disc = [c for c in features if c in BINARY + DISCRETE_ENGINEERED]

    if kind == "tree_native":
        return ColumnTransformer([("all", "passthrough", features)],
                                 verbose_feature_names_out=False)

    if kind == "tree_impute":
        return ColumnTransformer(
            [("cont", SimpleImputer(strategy="median"), cont),
             ("disc", SimpleImputer(strategy="most_frequent"), disc)],
            verbose_feature_names_out=False)

    if kind != "linear":
        raise ValueError(f"Unknown kind: {kind}")

    spline_cols = [c for c in ("RIDAGEYR", "BMXBMI") if splines and c in cont]
    other_cont = [c for c in cont if c not in spline_cols]

    def cont_pipe():
        steps = []
        if winsorize:
            steps.append(("winsor", Winsorizer()))
        steps.append(("impute", SimpleImputer(strategy="median")))
        return steps

    transformers = [
        ("cont", Pipeline(cont_pipe() + [("scale", StandardScaler())]),
         other_cont),
        ("disc", SimpleImputer(strategy="most_frequent"), disc),
    ]
    if spline_cols:
        transformers.append((
            "spline",
            Pipeline(cont_pipe() + [
                ("spline", SplineTransformer(n_knots=n_knots, degree=3,
                                             knots="quantile",
                                             include_bias=False)),
                ("scale", StandardScaler())]),
            spline_cols))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


# ---------------------------------------------------------------------------
# 4. Splits
# ---------------------------------------------------------------------------
def make_lockbox_split(train: pd.DataFrame, frac: float = 0.20,
                       seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified development / lockbox split. The lockbox is scored ONCE,
    at the very end, to estimate the test AUC."""
    dev, lock = train_test_split(train, test_size=frac,
                                 stratify=train[TARGET], random_state=seed)
    return dev.reset_index(drop=True), lock.reset_index(drop=True)


def get_cv(n_splits: int = 5, n_repeats: int = 3,
           seed: int = SEED) -> RepeatedStratifiedKFold:
    """The SAME fold object must be used for every model (paired comparison)."""
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                   random_state=seed)


# ---------------------------------------------------------------------------
# 5. Script entry point: process, save, sanity-check
# ---------------------------------------------------------------------------
def _sanity_check(dev: pd.DataFrame) -> None:
    """Quick CV run to prove the pipelines work end to end (not model search)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    cv = get_cv(n_repeats=1)
    y = dev[TARGET]
    checks = {
        "Logistic, age only": (
            ["RIDAGEYR"], Pipeline([
                ("prep", ColumnTransformer([("a", StandardScaler(), ["RIDAGEYR"])])),
                ("clf", LogisticRegression(max_iter=1000))])),
        "Logistic, raw features": (
            get_feature_list([]), Pipeline([
                ("prep", build_preprocessor(get_feature_list([]), "linear")),
                ("clf", LogisticRegression(max_iter=2000))])),
        "Logistic, all blocks + splines": (
            get_feature_list(), Pipeline([
                ("prep", build_preprocessor(get_feature_list(), "linear",
                                            splines=True)),
                ("clf", LogisticRegression(max_iter=5000, C=0.5))])),
        "HistGB, raw features": (
            get_feature_list([]), Pipeline([
                ("prep", build_preprocessor(get_feature_list([]), "tree_native")),
                ("clf", HistGradientBoostingClassifier(
                    learning_rate=0.05, max_leaf_nodes=15, random_state=SEED))])),
    }
    print("\nSanity check - 5-fold CV AUC on the development set:")
    for name, (cols, pipe) in checks.items():
        s = cross_val_score(pipe, dev[cols], y, cv=cv, scoring="roc_auc")
        print(f"  {name:<32s} {s.mean():.4f} (+/- {s.std(ddof=1):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=Path(__file__).resolve().parent,
                        help="Folder with the two CSV files")
    parser.add_argument("--out-dir", default=None,
                        help="Output folder (default: <data-dir>/processed)")
    parser.add_argument("--no-check", action="store_true",
                        help="Skip the quick CV sanity check")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    train, test = load_data(data_dir)
    print(f"Train: {train.shape}, Test: {test.shape}, "
          f"prevalence: {train[TARGET].mean():.3f}")

    summary = describe_data(train, test)
    print("\nData summary:\n", summary.to_string())
    summary.to_csv(out_dir / "data_summary.csv")

    feature_dupes = train.duplicated(RAW_FEATURES, keep=False).sum()
    print(f"\nRows sharing identical feature values: {feature_dupes} "
          f"(distinct ids; mostly young participants with all labs missing, "
          f"so coincidental rather than duplicated records - kept).")

    train_fe, test_fe = add_features(train), add_features(test)
    dev, lock = make_lockbox_split(train_fe)
    print(f"\nDevelopment set: {len(dev)} rows, lockbox: {len(lock)} rows "
          f"(prevalence {dev[TARGET].mean():.3f} / {lock[TARGET].mean():.3f})")

    train_fe.to_csv(out_dir / "train_features.csv", index=False)
    test_fe.to_csv(out_dir / "test_features.csv", index=False)
    pd.concat([dev[[ID_COL]].assign(set="dev"),
               lock[[ID_COL]].assign(set="lockbox")]).to_csv(
        out_dir / "split_ids.csv", index=False)
    print(f"Saved processed files to: {out_dir}")

    if not args.no_check:
        _sanity_check(dev)


if __name__ == "__main__":
    main()
