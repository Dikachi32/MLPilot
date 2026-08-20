"""Tests for evaluation engine."""
import pytest
import numpy as np
from ml.evaluation import EvaluationEngine

def test_evaluate_classification():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1])
    metrics = EvaluationEngine.evaluate_classification(y_true, y_pred)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "confusion_matrix" in metrics

def test_evaluate_regression():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
    metrics = EvaluationEngine.evaluate_regression(y_true, y_pred)
    assert "mse" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

def test_overfitting_detection():
    result = EvaluationEngine.detect_overfitting_underfitting(95, 70, "accuracy")
    assert result["status"] == "overfitting"
    result2 = EvaluationEngine.detect_overfitting_underfitting(45, 40, "accuracy")
    assert result2["status"] == "underfitting"
    result3 = EvaluationEngine.detect_overfitting_underfitting(85, 82, "accuracy")
    assert result3["status"] == "good_generalization"