"""Benchmark several regressors and train the production LightGBM models.

All models learn log(price). Benchmarks are scored with 5-fold CV on the
training split and once on the held-out test split. Run: python -m core.train
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    P10_MODEL_PATH,
    P90_MODEL_PATH,
    POINT_MODEL_PATH,
    TEST_PREDICTIONS_PATH,
)
from core.features import CATEGORICAL, FEATURE_LABELS, FEATURES, NUMERIC, TARGET, prepare

SEED = 42
LGBM_PARAMS = dict(
    n_estimators=900,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)
QUANTILE_PARAMS = dict(objective="quantile", n_estimators=500, min_child_samples=40)


def _sk_pipeline(regressor) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", regressor)])


def _lgbm(**overrides) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**{**LGBM_PARAMS, **overrides})


def _benchmarks() -> dict:
    return {
        "Linear Regression": lambda: _sk_pipeline(LinearRegression()),
        "Ridge Regression": lambda: _sk_pipeline(Ridge(alpha=1.0)),
        "Random Forest": lambda: _sk_pipeline(
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
        ),
        "Gradient Boosting": lambda: _sk_pipeline(
            GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, random_state=SEED)
        ),
        "LightGBM": _lgbm,
    }


def _scores(y_true_lakhs: np.ndarray, y_pred_lakhs: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true_lakhs, y_pred_lakhs)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_lakhs, y_pred_lakhs))),
        "R2": float(r2_score(y_true_lakhs, y_pred_lakhs)),
        "MAPE": float(mean_absolute_percentage_error(y_true_lakhs, y_pred_lakhs) * 100),
    }


def _fit_predict(factory, X_tr, y_tr_log, X_te) -> np.ndarray:
    model = factory()
    model.fit(X_tr, y_tr_log)
    return np.exp(model.predict(X_te))


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    X = prepare(df)
    y = df[TARGET].to_numpy()
    y_log = np.log(y)

    X_tr, X_te, y_tr, y_te, ylog_tr, _ = train_test_split(
        X, y, y_log, test_size=0.2, random_state=SEED, stratify=df["City"]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    results = []
    test_preds = {}
    for name, factory in _benchmarks().items():
        fold_scores = []
        for tr_idx, va_idx in kf.split(X_tr):
            pred = _fit_predict(factory, X_tr.iloc[tr_idx], ylog_tr[tr_idx], X_tr.iloc[va_idx])
            fold_scores.append(_scores(y_tr[va_idx], pred))
        cv = {f"CV_{k}": float(np.mean([s[k] for s in fold_scores])) for k in fold_scores[0]}
        cv["CV_MAPE_std"] = float(np.std([s["MAPE"] for s in fold_scores]))
        pred_te = _fit_predict(factory, X_tr, ylog_tr, X_te)
        test_preds[name] = pred_te
        results.append({"model": name, **cv, **{f"Test_{k}": v for k, v in _scores(y_te, pred_te).items()}})
        print(f"{name:20s} CV MAPE {cv['CV_MAPE']:.2f}%  Test MAPE {results[-1]['Test_MAPE']:.2f}%  R2 {results[-1]['Test_R2']:.4f}")

    best = min(results, key=lambda r: r["CV_MAPE"])["model"]
    if best != "LightGBM":
        print(f"NOTE: {best} beat LightGBM on CV MAPE; production still uses LightGBM for ranges and explanations.")

    # Production models: point estimate + P10/P90 quantiles. Raw quantile models
    # under-cover, so the interval is widened with conformalized quantile
    # regression (CQR) using a calibration split carved out of training data.
    point = _lgbm().fit(X_tr, ylog_tr)
    X_fit, X_cal, ylog_fit, ylog_cal = train_test_split(X_tr, ylog_tr, test_size=0.25, random_state=SEED)
    p10 = _lgbm(**QUANTILE_PARAMS, alpha=0.10).fit(X_fit, ylog_fit)
    p90 = _lgbm(**QUANTILE_PARAMS, alpha=0.90).fit(X_fit, ylog_fit)
    conformity = np.maximum(p10.predict(X_cal) - ylog_cal, ylog_cal - p90.predict(X_cal))
    n_cal = len(ylog_cal)
    cqr_margin = float(np.quantile(conformity, min(1.0, np.ceil((n_cal + 1) * 0.80) / n_cal)))

    lo = np.exp(p10.predict(X_te) - cqr_margin)
    hi = np.exp(p90.predict(X_te) + cqr_margin)
    coverage = float(np.mean((y_te >= lo) & (y_te <= hi)) * 100)
    print(f"Calibrated P10-P90 interval coverage on test: {coverage:.1f}% (target 80%), log margin {cqr_margin:.4f}")

    # Refit on all data for serving (the calibration margin carries over)
    point_full = _lgbm().fit(X, y_log)
    p10_full = _lgbm(**QUANTILE_PARAMS, alpha=0.10).fit(X, y_log)
    p90_full = _lgbm(**QUANTILE_PARAMS, alpha=0.90).fit(X, y_log)
    point_full.booster_.save_model(str(POINT_MODEL_PATH))
    p10_full.booster_.save_model(str(P10_MODEL_PATH))
    p90_full.booster_.save_model(str(P90_MODEL_PATH))

    # Mean |SHAP| on the test split (from the train-split model) as global importance
    contrib = point.predict(X_te, pred_contrib=True)[:, :-1]
    importance = dict(zip(FEATURES, np.abs(contrib).mean(axis=0)))
    importance = {FEATURE_LABELS[k]: float(v) for k, v in sorted(importance.items(), key=lambda kv: -kv[1])}

    te = df.loc[X_te.index, ["Listing_ID", "City", "Locality", "Property_Type", "Area_sqft", "BHK"]].copy()
    te["Actual"] = y_te
    for name, pred in test_preds.items():
        te[name] = np.round(pred, 2)
    te["P10"], te["P90"] = np.round(lo, 2), np.round(hi, 2)
    te.to_csv(TEST_PREDICTIONS_PATH, index=False)

    lgbm_row = next(r for r in results if r["model"] == "LightGBM")
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": int(len(df)),
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "best_cv_model": best,
        "production_model": "LightGBM",
        "production_test_mape": lgbm_row["Test_MAPE"],
        "production_test_r2": lgbm_row["Test_R2"],
        "interval_coverage_pct": coverage,
        "interval_log_margin": cqr_margin,
        "benchmarks": results,
        "feature_importance": importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved models and metrics to {MODEL_DIR}")


if __name__ == "__main__":
    main()
