"""Classification + separation metrics with stratified cross-validation."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import silhouette_score


def evaluate(X, y, seed=0, n_splits=5):
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y)
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = {}
    for name, clf in (("logreg", LogisticRegression(max_iter=2000)),
                      ("rf", RandomForestClassifier(n_estimators=300, random_state=seed))):
        pipe = make_pipeline(StandardScaler(), clf)
        acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        f1 = cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro")
        out[name] = {"accuracy": float(acc.mean()),
                     "accuracy_std": float(acc.std()),
                     "f1_macro": float(f1.mean())}
    try:
        sil = float(silhouette_score(StandardScaler().fit_transform(X), y))
    except Exception:
        sil = float("nan")
    out["silhouette"] = sil
    out["best_accuracy"] = max(out["logreg"]["accuracy"], out["rf"]["accuracy"])
    return out
