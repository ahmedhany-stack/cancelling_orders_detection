"""
utils/visualization.py
----------------------
Visualization utilities for Machine Learning projects.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

from sklearn.calibration import calibration_curve


# ==========================================================
# Target Distribution
# ==========================================================

def plot_target_distribution(df, target):

    plt.figure(figsize=(6, 4))

    sns.countplot(data=df, x=target)

    plt.title("Target Distribution")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Histogram
# ==========================================================

def plot_histogram(df, column):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[column],
        kde=True
    )

    plt.title(column)

    plt.tight_layout()

    plt.show()


# ==========================================================
# Boxplot
# ==========================================================

def plot_boxplot(df, column):

    plt.figure(figsize=(8, 4))

    sns.boxplot(x=df[column])

    plt.title(column)

    plt.tight_layout()

    plt.show()


# ==========================================================
# Correlation Heatmap
# ==========================================================

def plot_heatmap(df):

    plt.figure(figsize=(12, 8))

    sns.heatmap(
        df.corr(numeric_only=True),
        cmap="coolwarm",
        annot=False
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Missing Values
# ==========================================================

def plot_missing_values(df):

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=missing.index,
        y=missing.values
    )

    plt.xticks(rotation=90)

    plt.title("Missing Values")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Confusion Matrix
# ==========================================================

def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.show()


# ==========================================================
# ROC Curve
# ==========================================================

def plot_roc_curve(y_true, y_prob):

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.show()


# ==========================================================
# Precision Recall Curve
# ==========================================================

def plot_precision_recall_curve(
    y_true,
    y_prob
):

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        recall,
        precision,
        linewidth=2
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Feature Importance
# ==========================================================

def plot_feature_importance(
    model,
    feature_names,
    top_n=20
):

    importance = model.feature_importances_

    indices = np.argsort(
        importance
    )[::-1][:top_n]

    plt.figure(figsize=(10,6))

    sns.barplot(

        x=importance[indices],

        y=np.array(feature_names)[indices]

    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Prediction Distribution
# ==========================================================

def plot_prediction_distribution(
    y_prob
):

    plt.figure(figsize=(8,5))

    sns.histplot(
        y_prob,
        bins=30,
        kde=True
    )

    plt.xlabel("Predicted Probability")

    plt.title("Prediction Distribution")

    plt.tight_layout()

    plt.show()


# ==========================================================
# Calibration Curve
# ==========================================================

def plot_calibration_curve(
    y_true,
    y_prob
):

    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        prob_pred,
        prob_true,
        marker="o"
    )

    plt.plot(
        [0,1],
        [0,1],
        "--"
    )

    plt.xlabel("Mean Predicted Probability")

    plt.ylabel("Fraction of Positives")

    plt.title("Calibration Curve")

    plt.tight_layout()

    plt.show()