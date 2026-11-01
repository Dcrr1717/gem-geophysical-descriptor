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


# ---------------------------------------------------------------------------
# Grouped cross-validation (no leakage of correlated snapshots across folds)
# ---------------------------------------------------------------------------
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402


def _classifiers(seed):
    return (("logreg", LogisticRegression(max_iter=5000)),
            ("rf", RandomForestClassifier(n_estimators=300, random_state=seed,
                                          n_jobs=-1)))


def evaluate_grouped(X, y, groups, seed=0, n_splits=4, n_repeats=5):
    """Repeated StratifiedGroupKFold evaluation.

    All snapshots that share a group id (one simulated trajectory together with
    the surrogates derived from it) are always in the same fold, so temporally
    correlated snapshots can never be split between train and test.

    Returns the same summary keys as `evaluate` (per-classifier accuracy /
    accuracy_std / f1_macro, silhouette, best_accuracy) plus fold-level detail
    and the out-of-fold predictions of every repeat (`oof`) so that paired
    comparisons between feature sets can be bootstrapped. The folds depend
    only on (y, groups, seed), so every feature set sees identical splits.
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y)
    groups = np.asarray(groups)
    out = {}
    for name, clf in _classifiers(seed):
        fold_acc, rep_acc, rep_f1, oof_all = [], [], [], []
        for r in range(n_repeats):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed + r)
            oof = np.empty_like(y)
            for tr, te in cv.split(X, y, groups):
                pipe = make_pipeline(StandardScaler(), clf)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipe.fit(X[tr], y[tr])
                oof[te] = pipe.predict(X[te])
                fold_acc.append(accuracy_score(y[te], oof[te]))
            rep_acc.append(accuracy_score(y, oof))
            rep_f1.append(f1_score(y, oof, average="macro"))
            oof_all.append(oof)
        out[name] = {"accuracy": float(np.mean(rep_acc)),
                     "accuracy_std": float(np.std(fold_acc)),
                     "accuracy_std_repeats": float(np.std(rep_acc)),
                     "f1_macro": float(np.mean(rep_f1)),
                     "fold_accuracies": [float(a) for a in fold_acc],
                     "oof": [o.tolist() for o in oof_all]}
    try:
        sil = float(silhouette_score(StandardScaler().fit_transform(X), y))
    except Exception:
        sil = float("nan")
    out["silhouette"] = sil
    out["best_classifier"] = max(("logreg", "rf"), key=lambda k: out[k]["accuracy"])
    out["best_accuracy"] = out[out["best_classifier"]]["accuracy"]
    return out
