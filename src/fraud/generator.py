"""Synthetic transaction generator.

Produces realistic-looking card transactions with injected fraud signatures
(amount spikes / card-testing micro-charges, night-skewed timing, more foreign
and higher-distance activity, riskier merchant categories). No external data -
fully reproducible from a seed, and the fraud patterns are ours to control,
which is exactly what a streaming demo needs.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .config import Config, load_config

RISKY_CATEGORIES = {"online", "electronics", "travel", "atm"}


def _hour_weights(night: bool) -> np.ndarray:
    w = np.ones(24)
    for h in range(24):
        if night:
            w[h] = 3.0 if (h < 6 or h >= 22) else 0.6
        else:
            w[h] = 2.5 if 8 <= h <= 20 else 0.5
    return w / w.sum()


def _category_weights(categories: list[str], fraud: bool) -> np.ndarray:
    if fraud:
        w = np.array([3.0 if c in RISKY_CATEGORIES else 1.0 for c in categories], dtype=float)
    else:
        w = np.array([1.0 if c in RISKY_CATEGORIES else 1.5 for c in categories], dtype=float)
    return w / w.sum()


def generate_transactions(
    cfg: Config,
    n: int | None = None,
    seed: int | None = None,
    fraud_rate: float | None = None,
    amount_scale: float = 1.0,
    foreign_boost: float = 0.0,
) -> pd.DataFrame:
    """Generate ``n`` time-ordered transactions. Drift knobs shift the
    distribution (``amount_scale``, ``foreign_boost``, ``fraud_rate``)."""
    rng = np.random.default_rng(cfg.random_state if seed is None else seed)
    n = int(n or cfg.generator.n_transactions)
    fraud_rate = cfg.generator.fraud_rate if fraud_rate is None else fraud_rate
    categories = list(cfg.generator.merchant_categories)

    is_fraud = (rng.random(n) < fraud_rate).astype(int)
    timestamps = np.sort(rng.uniform(0, 30 * 24 * 3600, size=n))
    customer_ids = rng.integers(0, cfg.generator.n_customers, size=n)

    # "Stealth" fraud behaves exactly like legitimate activity, which caps the
    # achievable recall realistically, a fraud model that catches 100% is a
    # red flag, not a triumph. Only "obvious" fraud carries detectable signal.
    stealth = (rng.random(n) < 0.15) & (is_fraud == 1)
    obvious = (is_fraud == 1) & (~stealth)

    # Amounts overlap heavily; obvious fraud mixes card-testing micro-charges
    # with a heavier upper tail, but sits squarely on top of legitimate spend.
    pick = rng.random(n)
    amount = rng.lognormal(3.6, 0.95, n)
    amount = np.where(
        obvious,
        np.where(pick < 0.3, rng.uniform(0.5, 5.0, n), rng.lognormal(4.7, 0.9, n)),
        amount,
    ) * amount_scale

    hour = np.where(
        obvious,
        rng.choice(24, n, p=_hour_weights(night=True)),
        rng.choice(24, n, p=_hour_weights(night=False)),
    )

    p_obvious_foreign = min(0.35 + foreign_boost, 0.95)
    p_legit_foreign = min(0.07 + foreign_boost, 0.95)
    is_foreign = np.where(
        obvious, rng.random(n) < p_obvious_foreign, rng.random(n) < p_legit_foreign
    ).astype(int)

    distance = np.where(obvious, rng.gamma(2.0, 40, n), rng.gamma(2.0, 12, n))

    category = np.where(
        obvious,
        rng.choice(categories, n, p=_category_weights(categories, fraud=True)),
        rng.choice(categories, n, p=_category_weights(categories, fraud=False)),
    )

    return pd.DataFrame(
        {
            "transaction_id": np.arange(n),
            "customer_id": customer_ids,
            "timestamp": timestamps,
            "amount": np.round(amount, 2),
            "hour": hour.astype(int),
            "merchant_category": category,
            "is_foreign": is_foreign,
            "distance_from_home_km": np.round(distance, 1),
            "is_fraud": is_fraud,
        }
    )


def generate_stream_frame(cfg: Config, seed: int | None = None) -> pd.DataFrame:
    """Build a demo stream: a normal segment followed by a drifted segment
    (more fraud, larger amounts, more foreign) to exercise the drift monitor."""
    base_seed = cfg.random_state + 1 if seed is None else seed
    n_total = cfg.stream.demo_events
    cut = min(cfg.stream.drift_after, n_total)

    normal = generate_transactions(cfg, n=cut, seed=base_seed)
    drifted = generate_transactions(
        cfg,
        n=n_total - cut,
        seed=base_seed + 1,
        fraud_rate=min(cfg.generator.fraud_rate * 2.5, 0.5),
        amount_scale=1.6,
        foreign_boost=0.25,
    )
    combined = pd.concat([normal, drifted], ignore_index=True)
    # Monotonic timestamps so the stream is time-ordered across both segments.
    combined["timestamp"] = np.arange(n_total, dtype=float) * 45.0
    combined["transaction_id"] = np.arange(n_total)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and summarize synthetic transactions.")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    df = generate_transactions(cfg, n=args.n)
    print(f"Generated {len(df):,} transactions; fraud rate = {df['is_fraud'].mean():.4f}")
    print(df.head())


if __name__ == "__main__":
    main()
