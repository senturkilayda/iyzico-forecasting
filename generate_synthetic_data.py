#!/usr/bin/env python3
"""
generate_synthetic_data.py
--------------------------
Creates a small synthetic dataset that mimics the iyzico_data.csv schema.
Use this to verify the pipeline runs end-to-end without the real data.

Usage:
    python generate_synthetic_data.py           # writes to data/iyzico_data.csv
    python generate_synthetic_data.py --n 5     # 5 merchants
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)


def make_merchant(merchant_id, start, end, base_paid, noise_scale=0.25):
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    # Yearly seasonality (higher in Q4)
    doy = dates.dayofyear
    seasonal = 1 + 0.3 * np.sin(2 * np.pi * (doy - 90) / 365)

    # Weekly pattern (lower on weekends)
    weekly = np.where(dates.dayofweek < 5, 1.0, 0.35)

    # Upward trend
    trend = np.linspace(1.0, 1.5, n)

    # COVID dip (March–June 2020)
    covid_mask = (dates >= "2020-03-11") & (dates <= "2020-06-01")
    covid_factor = np.where(covid_mask, rng.uniform(0.4, 0.7), 1.0)

    # Noise
    noise = rng.lognormal(0, noise_scale, n)

    paid = base_paid * trend * seasonal * weekly * covid_factor * noise
    txns = np.maximum(1, (paid / rng.uniform(80, 200, n)).astype(int))

    df = pd.DataFrame({
        "merchant_id": merchant_id,
        "transaction_date": dates,
        "Total_Transaction": txns,
        "Total_Paid": np.round(paid, 2),
    })
    # Randomly drop ~5% of rows to simulate gaps
    drop = rng.random(n) < 0.05
    return df[~drop]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=7, help="Number of merchants")
    parser.add_argument("--out", default="data/iyzico_data.csv")
    args = parser.parse_args()

    starts = pd.date_range("2018-01-01", periods=args.n, freq="30D")
    bases = rng.uniform(5_000, 200_000, args.n)

    frames = []
    for i, (start, base) in enumerate(zip(starts, bases), 1):
        frames.append(make_merchant(i, start, "2020-09-30", base))

    data = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    data.to_csv(out, index=False)
    print(f"Written {len(data):,} rows → {out}")
    print(f"Merchants: {data['merchant_id'].nunique()}")
    print(f"Date range: {data['transaction_date'].min().date()} "
          f"→ {data['transaction_date'].max().date()}")


if __name__ == "__main__":
    main()
