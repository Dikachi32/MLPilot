"""Comprehensive model evaluation engine."""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

class EvaluationEngine:
    """Compute and store metrics for classification, regression, clustering, and NLP."""

    @staticmethod
    def evaluate_classification(y_true: Any, y_pred: Any, y_proba: Optional[Any] = None,
                                labels: Optional[Any] = None) -> Dict[str, Any]:
        avg = "binary" if len(np.unique(y_true)) == 2 else "weighted"
        metrics = {
            "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0) * 100, 2),
            "recall": round(recall_score(y_true, y_pred, average=avg, zero_division=0) * 100, 2),
            "f1": round(f1_score(y_true, y_pred, average=avg, zero_division=0) * 100, 2),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        }

        # ROC-AUC (binary or multiclass with proba)
        try:
            if y_proba is not None:
                if len(np.unique(y_true)) == 2:
                    metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba[:, 1]) * 100, 2)
                else:
                    metrics["roc_auc"] = round(roc_auc_score(y_true, y_proba, multi_class="ovr", average=avg) * 100, 2)
        except Exception:
            metrics["roc_auc"] = None

        # Log Loss
        try:
            if y_proba is not None:
                metrics["log_loss"] = round(log_loss(y_true, y_proba), 4)
        except Exception:
            metrics["log_loss"] = None

        return metrics

    @staticmethod
    def evaluate_regression(y_true: Any, y_pred: Any) -> Dict[str, float]:
        mse = mean_squared_error(y_true, y_pred)
        return {
            "mse": round(mse, 4),
            "mae": round(mean_absolute_error(y_true, y_pred), 4),
            "rmse": round(np.sqrt(mse), 4),
            "r2": round(r2_score(y_true, y_pred), 4),
        }

    @staticmethod
    def evaluate_clustering(X: Any, labels: Any) -> Dict[str, float]:
        if len(np.unique(labels)) < 2:
            return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
        return {
            "silhouette": round(silhouette_score(X, labels), 4),
            "davies_bouldin": round(davies_bouldin_score(X, labels), 4),
            "calinski_harabasz": round(calinski_harabasz_score(X, labels), 2),
        }

    @staticmethod
    def detect_overfitting_underfitting(train_score: float, test_score: float,
                                       metric_type: str = "accuracy") -> Dict[str, Any]:
        """Analyze train vs test gap to diagnose model behavior."""
        if metric_type in ("accuracy", "precision", "recall", "f1", "r2", "roc_auc"):
            gap = train_score - test_score
            if gap > 15:
                return {
                    "status": "overfitting",
                    "train_score": train_score,
                    "test_score": test_score,
                    "gap": round(gap, 2),
                    "recommendation": "Model memorizes training data. Try regularization, more data, simpler model, or PCA."
                }
            elif train_score < 60 and test_score < 60:
                return {
                    "status": "underfitting",
                    "train_score": train_score,
                    "test_score": test_score,
                    "gap": round(gap, 2),
                    "recommendation": "Model too simple. Try more features, ensemble methods, or hyperparameter tuning."
                }
            else:
                return {
                    "status": "good_generalization",
                    "train_score": train_score,
                    "test_score": test_score,
                    "gap": round(gap, 2),
                    "recommendation": "Model generalizes well."
                }
        else:
            # For error metrics (MSE, MAE, RMSE) lower is better
            gap = test_score - train_score
            if gap > train_score * 0.5 and train_score > 0:
                return {
                    "status": "overfitting",
                    "train_score": train_score,
                    "test_score": test_score,
                    "gap": round(gap, 2),
                    "recommendation": "Model overfits. Consider regularization, simpler model, or cross-validation."
                }
            return {
                "status": "good_generalization",
                "train_score": train_score,
                "test_score": test_score,
                "gap": round(gap, 2),
                "recommendation": "Model generalizes well."
            }