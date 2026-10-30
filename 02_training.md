# Baseline XGBoost Training

## Training configuration

| Setting | Value |
|---|---:|
| Instance type | `ml.m5.large` |
| Training rounds | 80 |
| Learning rate (`eta`) | 0.15 |
| Maximum tree depth | 4 |
| Subsample ratio | 0.85 |

## Results

| Metric | Value |
|---|---:|
| Best validation AUC | 0.74700 (round 18) |
| Final training AUC | 0.96054 |
| Final validation AUC | 0.72310 |
| Billable training time | 129 seconds |

**Finding:** The gap between training and validation AUC indicates substantial overfitting. Consider early stopping and hyperparameter optimization.