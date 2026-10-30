"""Shared settings and cautious state persistence for SageMaker capstone."""

import json
import os
from pathlib import Path
import boto3
from dotenv import load_dotenv
from sagemaker.core import image_uris

# Load settings from the .env file beside this script (not from PowerShell exports).
# override=True makes the .env file take precedence over existing shell values.
load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / ".lab-state.json"


def env():
    region = os.environ.get("AWS_REGION", "us-east-1")
    session = boto3.Session(region_name=region)
    account = session.client("sts").get_caller_identity()["Account"]
    bucket = os.environ.get("CAPSTONE_BUCKET", f"sagemaker-capstone-{account}-{region}")
    role = os.environ.get("SAGEMAKER_ROLE_ARN", "")
    instance = os.environ.get("CAPSTONE_INSTANCE", "ml.m5.large")
    return session, region, account, bucket, role, instance


def require_role(role):
    if not role.startswith("arn:aws:iam::") or "<" in role:
        raise SystemExit(
            "Set SAGEMAKER_ROLE_ARN to your REAL existing SageMaker execution role ARN."
        )


def get_state():
    return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}


def save_state(**updates):
    s = get_state()
    s.update(updates)
    STATE_FILE.write_text(json.dumps(s, indent=2) + "\n")
    return s


def xgboost_image(region):
    # Pin the algorithm version; SageMaker Python SDK v2 resolves the regional ECR image.

    return image_uris.retrieve("xgboost", region, version="1.7-1")


def channels(bucket):
    out = []
    for name in ("train", "validation"):
        out.append(
            {
                "ChannelName": name,
                "ContentType": "text/csv",
                "InputMode": "File",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket}/processed/{name}/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        )
    return out


def wait_training(sm, name):
    sm.get_waiter("training_job_completed_or_stopped").wait(
        TrainingJobName=name, WaiterConfig={"Delay": 30, "MaxAttempts": 60}
    )
    out = sm.describe_training_job(TrainingJobName=name)
    if out["TrainingJobStatus"] != "Completed":
        raise RuntimeError(
            f"Training not completed: {out['TrainingJobStatus']} / {out.get('FailureReason')}"
        )
    return out
