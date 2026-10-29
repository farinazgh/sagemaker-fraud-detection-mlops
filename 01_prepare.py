"""Generate reproducible synthetic transactions and upload XGBoost CSV datasets."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from shared import env, save_state


def main():
    session, region, account, bucket, role, instance = env()
    s3 = session.client("s3")
    try:
        s3.head_bucket(Bucket=bucket)
        print("Using accessible bucket:", bucket)
    except Exception:
        params = {"Bucket": bucket}
        if region != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": region}
        s3.create_bucket(**params)
        s3.get_waiter("bucket_exists").wait(Bucket=bucket)
        print("Created bucket:", bucket)
    n = 5000
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "amount": np.clip(rng.lognormal(4.0, 0.9, n), 1, 1500),
            "hour": rng.integers(0, 24, n),
            "merchant_risk": rng.binomial(1, 0.18, n),
            "transactions_24h": rng.poisson(2.1, n),
            "account_age_days": rng.integers(1, 1800, n),
            "country_mismatch": rng.binomial(1, 0.12, n),
            "device_trust": rng.uniform(0, 1, n),
            "prior_chargebacks": rng.poisson(0.15, n),
        }
    )
    logit = (
        -4.5
        + 0.003 * frame["amount"]
        + 0.95 * frame["merchant_risk"]
        + 0.24 * frame["transactions_24h"]
        + 1.6 * frame["country_mismatch"]
        + 1.3 * (1 - frame["device_trust"])
        + 0.8 * frame["prior_chargebacks"]
        - 0.0005 * frame["account_age_days"]
    )
    probability = 1 / (1 + np.exp(-logit))
    frame.insert(0, "is_fraud", rng.binomial(1, probability))
    train, test = train_test_split(
        frame, test_size=0.20, random_state=42, stratify=frame["is_fraud"]
    )
    train, validation = train_test_split(
        train, test_size=0.25, random_state=42, stratify=train["is_fraud"]
    )
    local = Path("artifacts")
    local.mkdir(exist_ok=True)
    raw = local / "transactions.csv"
    frame.to_csv(raw, index=False)
    s3.upload_file(str(raw), bucket, "raw/transactions.csv")
    for key, data in [("train", train), ("validation", validation), ("test", test)]:
        path = local / f"{key}.csv"
        data.to_csv(path, index=False, header=False, float_format="%.8f")
        s3.upload_file(str(path), bucket, f"processed/{key}/{key}.csv")
        print(f"{key}: {len(data)} rows -> s3://{bucket}/processed/{key}/")
    save_state(
        bucket=bucket,
        region=region,
        account=account,
        feature_names=frame.columns[1:].tolist(),
    )
    print("Positive label proportion:", round(float(frame.is_fraud.mean()), 4))
    print("S3 raw dataset:", f"s3://{bucket}/raw/transactions.csv")


if __name__ == "__main__":
    main()
