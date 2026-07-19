"""
utils/metrics.py
----------------
Utility functions for evaluating Machine Learning models.

Author: Ahmed Hany
"""

import json
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# Classification Metrics
# ==========================================================

def classification_metrics(
    y_true,
    y_pred,
    y_prob=None,
    average="binary"
):
    """
    Calculate classification metrics.

    Returns
    -------
    dict
    """

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        ),
        "Recall": recall_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_true,
            y_pred,
            average=average,
            zero_division=0
        ),
    }

    if y_prob is not None:

        metrics["ROC AUC"] = roc_auc_score(
            y_true,
            y_prob
        )

        metrics["PR AUC"] = average_precision_score(
            y_true,
            y_prob
        )

    return metrics


# ==========================================================
# Regression Metrics
# ==========================================================

def regression_metrics(
    y_true,
    y_pred
):
    """
    Calculate regression metrics.
    """

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    metrics = {

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        ),

        "MSE": mse,

        "RMSE": np.sqrt(mse),

        "R2 Score": r2_score(
            y_true,
            y_pred
        )

    }

    return metrics


# ==========================================================
# Confusion Matrix
# ==========================================================

def get_confusion_matrix(
    y_true,
    y_pred
):
    """
    Return confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )


# ==========================================================
# Classification Report
# ==========================================================

def get_classification_report(
    y_true,
    y_pred
):
    """
    Return classification report.
    """

    return classification_report(
        y_true,
        y_pred
    )


# ==========================================================
# Specificity
# ==========================================================

def specificity(
    y_true,
    y_pred
):
    """
    True Negative Rate.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tn / (tn + fp)


# ==========================================================
# Sensitivity
# ==========================================================

def sensitivity(
    y_true,
    y_pred
):
    """
    True Positive Rate.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    return tp / (tp + fn)


# ==========================================================
# Threshold Search
# ==========================================================

def threshold_search(
    y_true,
    y_prob,
    start=0.10,
    end=0.95,
    step=0.01
):
    """
    Search for best threshold using F1 Score.
    """

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(
        start,
        end,
        step
    ):

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            y_pred
        )

        if score > best_f1:

            best_f1 = score

            best_threshold = threshold

    return best_threshold, best_f1


# ==========================================================
# Compare Models
# ==========================================================

def compare_models(
    results
):
    """
    Print multiple model results.

    Parameters
    ----------
    results : dict

    Example

    {
        "Random Forest": {...},
        "XGBoost": {...}
    }
    """

    print("=" * 70)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")

        print("-" * 70)

        for key, value in metrics.items():

            print(
                f"{key:<15}: {value:.4f}"
            )

    print("=" * 70)


# ==========================================================
# Print Metrics
# ==========================================================

def print_metrics(
    metrics
):
    """
    Pretty print metrics.
    """

    print("=" * 50)

    for key, value in metrics.items():

        print(
            f"{key:<15}: {value:.4f}"
        )

    print("=" * 50)


# ==========================================================
# Save Metrics
# ==========================================================

def save_metrics(
    metrics,
    path
):
    """
    Save metrics as JSON.
    """

    with open(
        path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )


# ==========================================================
# Load Metrics
# ==========================================================

def load_metrics(
    path
):
    """
    Load metrics JSON.
    """

    with open(path) as f:

        return json.load(f)