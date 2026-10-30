"""Create one short-lived baseline training job with the built-in XGBoost algorithm."""

from datetime import datetime, timezone

from shared import env, require_role, channels, save_state, wait_training, xgboost_image


def main():
    session, region, account, bucket, role, instance = env()
    require_role(role)
    sagemaker_client = session.client("sagemaker")
    name = "fraud-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    # Submit a baseline XGBoost training job with fixed hyperparameters, processed
    # train/validation inputs, and S3 output so later workflow steps can use its model.
    sagemaker_client.create_training_job(
        TrainingJobName=name,
        AlgorithmSpecification={
            "TrainingImage": xgboost_image(region),
            "TrainingInputMode": "File",
        },
        RoleArn=role,
        InputDataConfig=channels(bucket),
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/model-artifacts/"},
        ResourceConfig={
            "InstanceType": instance,
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 900},
        HyperParameters={
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "num_round": "80",
            "max_depth": "4",
            "eta": "0.15",
            "subsample": "0.85",
            "early_stopping_rounds": "10",
        },
        Tags=[{"Key": "Project", "Value": "SageMakerCapstone"}],
    )
    save_state(baseline_job=name)
    print("Started training:", name)
    job = wait_training(sagemaker_client, name)
    model_artifact = job["ModelArtifacts"]["S3ModelArtifacts"]
    save_state(baseline_artifact=model_artifact)
    print("Completed. Model:", model_artifact)


if __name__ == "__main__":
    main()
