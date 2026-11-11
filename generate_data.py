"""Generate a synthetic credit-card transaction dataset (data/transactions.csv).

Fraud probability is driven by realistic risk factors (night-time, large amount
relative to the customer's average, far from home, card-not-present,
international, bursts of transactions, risky merchant categories) so the
ensemble models have a real signal to learn.
"""
import numpy as np
import pandas as pd

CATEGORIES = ["grocery", "fuel", "restaurant", "travel", "electronics",
              "online_shopping", "entertainment", "health", "jewellery", "atm_cash"]
CAT_RISK = {"grocery": -0.6, "fuel": -0.3, "restaurant": -0.4, "travel": 0.5,
            "electronics": 0.9, "online_shopping": 0.8, "entertainment": 0.0,
            "health": -0.7, "jewellery": 1.1, "atm_cash": 0.4}


def generate(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    cat = rng.choice(CATEGORIES, n, p=[.22, .12, .14, .06, .08, .14, .07, .06, .03, .08])
    hour = np.where(rng.random(n) < 0.12, rng.integers(0, 5, n), rng.integers(6, 24, n))
    dow = rng.integers(0, 7, n)
    avg_amt = rng.lognormal(3.9, 0.45, n)                      # customer's 30-day average
    amount = avg_amt * rng.lognormal(0, 0.55, n)
    big = rng.random(n) < 0.04
    amount = np.where(big, amount * rng.uniform(4, 12, n), amount)
    dist_home = np.abs(rng.normal(8, 12, n)) * np.where(rng.random(n) < 0.07, rng.uniform(8, 40, n), 1)
    dist_last = np.abs(rng.normal(5, 9, n)) * np.where(rng.random(n) < 0.05, rng.uniform(10, 60, n), 1)
    txn24 = rng.poisson(2.2, n) + np.where(rng.random(n) < 0.05, rng.integers(5, 14, n), 0)
    card_present = (rng.random(n) < np.where(np.isin(cat, ["online_shopping"]), 0.0, 0.8)).astype(int)
    intl = (rng.random(n) < 0.06).astype(int)

    ratio = amount / avg_amt
    z = (-6.6
         + 2.2 * np.log1p(np.maximum(ratio - 1.5, 0))
         + 2.2 * (hour < 5)
         + 0.04 * np.minimum(dist_home, 120)
         + 0.025 * np.minimum(dist_last, 150)
         + 0.45 * np.maximum(txn24 - 5, 0)
         + 1.6 * (1 - card_present)
         + 1.8 * intl
         + 0.0012 * np.minimum(amount, 3000)
         + np.array([CAT_RISK[c] for c in cat])
         + rng.normal(0, 0.35, n))
    p = 1 / (1 + np.exp(-z))
    fraud = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "amount": amount.round(2), "hour": hour, "day_of_week": dow,
        "merchant_category": cat, "distance_from_home_km": dist_home.round(1),
        "distance_from_last_txn_km": dist_last.round(1), "txn_count_24h": txn24,
        "avg_amount_30d": avg_amt.round(2), "card_present": card_present,
        "international": intl, "is_fraud": fraud})


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/transactions.csv", index=False)
    print(df.shape, "fraud rate: %.2f%%" % (100 * df.is_fraud.mean()))
