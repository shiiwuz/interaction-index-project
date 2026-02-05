#!/usr/bin/env python3
"""Calibrate model predictions with a monotonic mapping (isotonic regression).

Fits on validation set: y_true ~ f(y_pred), where f is monotone non-decreasing.
Then applies to test set and reports metric changes.

Inputs:
- Feature cache: outputs/pref_xgb_fe_cache_seed42.npz
- Trained XGBoost model: outputs/pref_xgb_fe_report.xgb.json
- Report JSON: outputs/pref_xgb_fe_report.json (for best_iteration)

Outputs:
- outputs/pref_xgb_fe_calibration_report.json
- outputs/pref_xgb_fe_test_dist_calibrated.png
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from sklearn.isotonic import IsotonicRegression


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb))
    if denom == 0:
        return float("nan")
    return float((ra @ rb) / denom)


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    err = (yhat - y).astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    sp = spearman(y, yhat)
    return {"mae": mae, "rmse": rmse, "spearman": sp, "std_pred": float(np.std(yhat))}


def main() -> int:
    cache = Path("outputs/pref_xgb_fe_cache_seed42.npz")
    rep_path = Path("outputs/pref_xgb_fe_report.json")

    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    model_path = Path(rep["model"])
    best_iter = (rep.get("report") or {}).get("best", {}).get("iteration")

    z = np.load(cache, allow_pickle=False)
    X = np.hstack([z["X_title"], z["X_sum"], z["X_dom"], z["meta"]]).astype(np.float32)
    y = z["y"].astype(np.float32)

    n = len(y)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    sl_val = slice(n_train, n_train + n_val)
    sl_test = slice(n_train + n_val, n)

    booster = xgb.Booster()
    booster.load_model(model_path)

    kwargs = {}
    if best_iter is not None:
        kwargs["iteration_range"] = (0, int(best_iter) + 1)

    yhat_val = booster.predict(xgb.DMatrix(X[sl_val]), **kwargs).astype(np.float32)
    y_val = y[sl_val]

    yhat_test = booster.predict(xgb.DMatrix(X[sl_test]), **kwargs).astype(np.float32)
    y_test = y[sl_test]

    # Fit monotone mapping on val.
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(yhat_val, y_val)

    yhat_val_cal = iso.predict(yhat_val).astype(np.float32)
    yhat_test_cal = iso.predict(yhat_test).astype(np.float32)

    out = {
        "best_iteration": best_iter,
        "metrics": {
            "val_before": metrics(y_val, yhat_val),
            "val_after": metrics(y_val, yhat_val_cal),
            "test_before": metrics(y_test, yhat_test),
            "test_after": metrics(y_test, yhat_test_cal),
        },
        "note": "IsotonicRegression fitted on val (pred->true) and applied to test.",
    }

    Path("outputs/pref_xgb_fe_calibration_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Plot: distribution + QQ, before/after.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def qq(ax, a, b, title):
        qs = np.linspace(0.01, 0.99, 99)
        qa = np.quantile(a, qs)
        qb = np.quantile(b, qs)
        ax.plot(qa, qb, lw=2)
        lo = min(qa.min(), qb.min())
        hi = max(qa.max(), qb.max())
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
        ax.set_title(title)
        ax.set_xlabel("true quantiles")
        ax.set_ylabel("pred quantiles")

    fig = plt.figure(figsize=(14, 8), dpi=160)

    ax1 = fig.add_subplot(2, 2, 1)
    bins = np.linspace(min(y_test.min(), yhat_test.min(), yhat_test_cal.min()), max(y_test.max(), yhat_test.max(), yhat_test_cal.max()), 45)
    ax1.hist(y_test, bins=bins, alpha=0.45, density=True, label="true (test)")
    ax1.hist(yhat_test, bins=bins, alpha=0.45, density=True, label="pred before")
    ax1.hist(yhat_test_cal, bins=bins, alpha=0.45, density=True, label="pred after")
    ax1.set_title("Test distribution (log-space)")
    ax1.set_xlabel("y")
    ax1.set_ylabel("density")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    qq(ax2, y_test, yhat_test, "Q-Q before")

    ax3 = fig.add_subplot(2, 2, 3)
    qq(ax3, y_test, yhat_test_cal, "Q-Q after (isotonic)")

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(yhat_val, y_val, s=4, alpha=0.15)
    xs = np.linspace(float(np.min(yhat_val)), float(np.max(yhat_val)), 200)
    ax4.plot(xs, iso.predict(xs), color="red", lw=2)
    ax4.set_title("Calibration curve (val): y_true vs y_pred")
    ax4.set_xlabel("y_pred")
    ax4.set_ylabel("y_true")

    fig.suptitle("Isotonic calibration on val; evaluated on test", fontsize=10)
    fig.tight_layout()
    fig.savefig("outputs/pref_xgb_fe_test_dist_calibrated.png")

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote outputs/pref_xgb_fe_calibration_report.json")
    print("wrote outputs/pref_xgb_fe_test_dist_calibrated.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
