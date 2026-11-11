"""Training + inference code for the ensemble fraud detector.

Models (Unit 5 - Combining Models):
  CART                 - single classification tree (base learner / baseline)
  Random Forest        - bagging of trees (variance reduction)
  AdaBoost             - boosting with decision stumps (adaptive basis function model)
  Gradient Boosting    - boosting with gradient descent in function space
  Voting Classifier    - soft-voting ensemble of the four models above
  BMA                  - Bayesian-model-averaging style weighted ensemble
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "transactions.csv")
MODEL_PATH = os.path.join(BASE, "models", "ensemble.joblib")
METRICS_PATH = os.path.join(BASE, "models", "metrics.json")

CATEGORIES = ["grocery", "fuel", "restaurant", "travel", "electronics",
              "online_shopping", "entertainment", "health", "jewellery", "atm_cash"]
NUMERIC = ["amount", "hour", "day_of_week", "distance_from_home_km",
           "distance_from_last_txn_km", "txn_count_24h", "avg_amount_30d",
           "card_present", "international"]


def featurize(df):
    """Raw transaction rows -> numeric feature matrix (fixed column order)."""
    X = df[NUMERIC].astype(float).copy()
    X["amount_ratio"] = X["amount"] / X["avg_amount_30d"].clip(lower=1)
    X["is_night"] = (X["hour"] < 5).astype(float)
    for c in CATEGORIES:
        X["cat_" + c] = (df["merchant_category"] == c).astype(float)
    return X


def _metrics(y, pred, proba):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "confusion": confusion_matrix(y, pred).tolist(),
    }


def train():
    df = pd.read_csv(DATA)
    X, y = featurize(df), df["is_fraud"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=7)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=7)
    sw = compute_sample_weight("balanced", y_tr)

    models = {
        "CART": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=1),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=1),
        "AdaBoost": AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, class_weight="balanced"),
                                       n_estimators=150, learning_rate=0.6, random_state=1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08,
                                                        max_depth=3, subsample=0.8, random_state=1),
    }
    # Gradient boosting has no class_weight argument -> pass sample weights
    for name, m in models.items():
        if name == "Gradient Boosting":
            m.fit(X_tr, y_tr, sample_weight=sw)
        else:
            m.fit(X_tr, y_tr)

    # --- Bayesian model averaging: weight_k  ∝  exp(-n_eff * logloss_k(validation)) ---
    # (tempered marginal-likelihood approximation using held-out data)
    n_eff = 40
    ll = {k: log_loss(y_val, np.clip(m.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6)) for k, m in models.items()}
    raw = np.array([np.exp(-n_eff * (v - min(ll.values()))) for v in ll.values()])
    bma_w = dict(zip(models.keys(), (raw / raw.sum()).tolist()))

    def bma_proba(Xm):
        return sum(bma_w[k] * m.predict_proba(Xm)[:, 1] for k, m in models.items())

    results, roc = {}, {}
    all_models = dict(models)
    for name, m in all_models.items():
        p = m.predict_proba(X_te)[:, 1]
        results[name] = _metrics(y_te, (p >= 0.5).astype(int), p)
        fpr, tpr, _ = roc_curve(y_te, p)
        idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
        roc[name] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
    # soft voting = average of the four models' class probabilities (what VotingClassifier(voting='soft') does)
    pv = np.mean([m.predict_proba(X_te)[:, 1] for m in models.values()], axis=0)
    results["Voting Classifier"] = _metrics(y_te, (pv >= 0.5).astype(int), pv)
    fpr, tpr, _ = roc_curve(y_te, pv)
    idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    roc["Voting Classifier"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}
    pb = bma_proba(X_te)
    results["BMA Ensemble"] = _metrics(y_te, (pb >= 0.5).astype(int), pb)

    # insights for the dashboard charts
    df["is_fraud"] = df["is_fraud"].astype(int)
    by_hour = df.groupby("hour")["is_fraud"].mean().reindex(range(24), fill_value=0) * 100
    by_cat = (df.groupby("merchant_category")["is_fraud"].mean() * 100).sort_values(ascending=False)
    imp = dict(zip(X.columns, models["Random Forest"].feature_importances_))
    imp = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:10])

    info = {
        "dataset": {"rows": int(len(df)), "fraud": int(df.is_fraud.sum()),
                    "fraud_rate": float(df.is_fraud.mean() * 100),
                    "train": int(len(X_tr)), "test": int(len(X_te))},
        "results": results, "roc": roc, "bma_weights": bma_w,
        "by_hour": by_hour.round(2).tolist(),
        "by_category": {"labels": by_cat.index.tolist(), "values": by_cat.round(2).tolist()},
        "importance": {"labels": list(imp.keys()), "values": [round(float(v), 4) for v in imp.values()]},
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"models": models, "bma_w": bma_w, "columns": list(X.columns)}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(info, f)
    return info


_bundle = None


def load():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            train()
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict(row: dict):
    """Return per-model fraud probabilities and the final ensemble verdict."""
    b = load()
    df = pd.DataFrame([row])
    X = featurize(df)[b["columns"]]
    probs = {k: float(m.predict_proba(X)[0, 1]) for k, m in b["models"].items()}
    vote = float(np.mean(list(probs.values())))                      # soft voting
    bma = float(sum(b["bma_w"][k] * p for k, p in probs.items()))    # BMA
    probs["Voting Classifier"] = vote
    probs["BMA Ensemble"] = bma
    votes_fraud = sum(1 for k in b["models"] if probs[k] >= 0.5)
    risk = "HIGH" if vote >= 0.7 else "MEDIUM" if vote >= 0.5 else "LOW"
    return {"probabilities": probs, "final_probability": vote, "is_fraud": vote >= 0.5,
            "risk": risk, "models_voting_fraud": votes_fraud}


if __name__ == "__main__":
    info = train()
    print(f"{'Model':<20}{'Acc':>8}{'Prec':>8}{'Rec':>8}{'F1':>8}{'AUC':>8}")
    for k, v in info["results"].items():
        print(f"{k:<20}{v['accuracy']:8.3f}{v['precision']:8.3f}{v['recall']:8.3f}{v['f1']:8.3f}{v['roc_auc']:8.3f}")
    print("BMA weights:", {k: round(v, 3) for k, v in info["bma_weights"].items()})
