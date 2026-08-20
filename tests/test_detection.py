"""Tests for dataset detection engine."""
import pytest
import pandas as pd
import numpy as np
from ml.detection import analyze_dataset, detect_problem_type, suggest_target_columns, recommend_algorithms

def test_analyze_dataset():
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "cat1": ["a", "b", "a", "b", "a"],
        "text1": ["hello world", "foo bar", "baz qux", "hello again", "test text"],
        "date1": pd.date_range("2020-01-01", periods=5),
        "constant": [1, 1, 1, 1, 1],
    })
    analysis = analyze_dataset(df)
    assert analysis["n_rows"] == 5
    assert analysis["n_cols"] == 5
    assert "num1" in analysis["numeric_columns"]
    assert "cat1" in analysis["categorical_columns"]
    assert "text1" in analysis["text_columns"]
    assert "date1" in analysis["datetime_columns"]
    assert "constant" in analysis["constant_columns"]

def test_detect_classification():
    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    result = detect_problem_type(df, target_col="target")
    assert result["problem_type"] == "classification"
    assert result["confidence"] == "high"

def test_detect_regression():
    df = pd.DataFrame({
        "feat1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [1.5, 2.3, 3.1, 4.8, 5.2, 6.1, 7.4, 8.9, 9.2, 10.5],
    })
    result = detect_problem_type(df, target_col="target")
    assert result["problem_type"] == "regression"

def test_detect_unsupervised_no_target():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [5, 4, 3, 2, 1],
    })
    result = detect_problem_type(df)
    assert result["problem_type"] == "unsupervised"

def test_recommend_algorithms():
    assert "Random Forest" in recommend_algorithms({"problem_type": "classification"})
    assert "Linear Regression" in recommend_algorithms({"problem_type": "regression"})
    assert "K-Means" in recommend_algorithms({"problem_type": "unsupervised"})