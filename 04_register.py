"""Register best XGBoost model as PendingManualApproval (do NOT autoapprove)."""

from botocore.exceptions import ClientError

from shared import env, require_role, get_state, save_state, xgboost_image


def main():
    session, region, account, bucket, role, instance = env()
    require_role(role)
    sagemaker_client = session.client("sagemaker")
    state = get_state()
    artifact = state.get("best_model_artifact") or state.get("baseline_artifact")
    if not artifact:
        raise SystemExit("First run 02_train.py or 03_tune.py.")
    group = "fraud-capstone-models"
    try:
        sagemaker_client.describe_model_package_group(ModelPackageGroupName=group)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in (
            "ValidationException",
            "ResourceNotFound",
        ):
            raise
        sagemaker_client.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="Synthetic fraud scoring capstone",
        )
    # Register the selected S3 model artifact with the XGBoost inference image and
    # declare the input/output formats and supported deployment instance types.
    # PendingManualApproval prevents downstream deployment from treating the model
    # as approved before a person has reviewed its evaluation and artifact.
    result = sagemaker_client.create_model_package(
        ModelPackageGroupName=group,
        ModelPackageDescription="XGBoost synthetic fraud risk classifier",
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [{"Image": xgboost_image(region), "ModelDataUrl": artifact}],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
            "SupportedTransformInstanceTypes": ["ml.m5.large", "ml.m5.xlarge"],
        },
    )
    package_arn = result["ModelPackageArn"]
    save_state(model_package_group=group, model_package_arn=package_arn)
    print("Registered (pending manual approval):", package_arn)
    print("Approve when you reviewed evaluation and artifact:")
    print(
        f"aws sagemaker update-model-package --model-package-arn {package_arn} --model-approval-status Approved --region {region}"
    )


if __name__ == "__main__":
    main()
