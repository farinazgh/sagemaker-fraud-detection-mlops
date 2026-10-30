"""Launch just three SEQUENTIAL HPO trials; inspect best metric and artifact."""

import time
from datetime import datetime, timezone

from shared import env, require_role, channels, save_state, xgboost_image


def main():
    session, region, account, bucket, role, instance = env()
    require_role(role)

    sagemaker_client = session.client("sagemaker")

    name = "fraud-hpo-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    # Limit the search to three trials and run them one at a time.
    sagemaker_client.create_hyper_parameter_tuning_job(
        HyperParameterTuningJobName=name,
        HyperParameterTuningJobConfig={
            "Strategy": "Bayesian",
            # Select the trial with the highest validation AUC.
            "HyperParameterTuningJobObjective": {
                "Type": "Maximize",
                "MetricName": "validation:auc",
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": 3,
                "MaxParallelTrainingJobs": 1,
            },
            "ParameterRanges": {
                "IntegerParameterRanges": [
                    {"Name": "max_depth", "MinValue": "2", "MaxValue": "6"}
                ],
                "ContinuousParameterRanges": [
                    {"Name": "eta", "MinValue": "0.05", "MaxValue": "0.30"}
                ],
            },
            "TrainingJobEarlyStoppingType": "Off",
        },
        TrainingJobDefinition={
            # Keep these training settings fixed while SageMaker tunes depth and learning rate.
            "StaticHyperParameters": {
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "num_round": "80",
                "subsample": "0.85",
            },
            "AlgorithmSpecification": {
                "TrainingImage": xgboost_image(region),
                "TrainingInputMode": "File",
            },
            "RoleArn": role,
            "InputDataConfig": channels(bucket),
            "OutputDataConfig": {"S3OutputPath": f"s3://{bucket}/model-artifacts/"},
            "ResourceConfig": {
                "InstanceType": instance,
                "InstanceCount": 1,
                "VolumeSizeInGB": 10,
            },
            "StoppingCondition": {"MaxRuntimeInSeconds": 900},
        },
        Tags=[{"Key": "Project", "Value": "SageMakerCapstone"}],
    )
    save_state(hpo_job=name)
    print("HPO launched:", name)
    # Poll until SageMaker reports a terminal tuning-job status.
    while True:
        result = sagemaker_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=name
        )
        status = result["HyperParameterTuningJobStatus"]
        print(
            "Tuning status:",
            status,
            "completed:",
            result.get("TrainingJobStatusCounters", {}).get("Completed", 0),
        )
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(45)
    if status != "Completed":
        raise RuntimeError(
            f'HPO status: {status}; {result.get("FailureReason", "No reason supplied")}'
        )
    best = result["BestTrainingJob"]
    desc = sagemaker_client.describe_training_job(
        TrainingJobName=best["TrainingJobName"]
    )
    # Persist the selected trial and its model artifact for downstream scripts.
    save_state(
        best_training_job=best["TrainingJobName"],
        best_model_artifact=desc["ModelArtifacts"]["S3ModelArtifacts"],
    )
    print("Best job:", best["TrainingJobName"])
    print("Best metric:", best.get("FinalHyperParameterTuningJobObjectiveMetric"))
    print("Selected hyperparameters:", best.get("TunedHyperParameters"))
    print("Artifact:", desc["ModelArtifacts"]["S3ModelArtifacts"])


if __name__ == "__main__":
    main()
