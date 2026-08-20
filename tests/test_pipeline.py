"""Integration tests for the AutoML pipeline."""
import pytest
import pandas as pd
import numpy as np
from ml.pipeline import MLPilotPipeline

def test_classification_pipeline():
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "feat3": np.random.choice(["a", "b", "c"], 100),
        "target": np.random.choice([0, 1], 100),
    })
    pipeline = MLPilotPipeline(artifact_folder="test_artifacts")
    result = pipeline.run(df, target_col="target", user_plan="trial", optimize=False)
    assert result["status"] == "completed"
    assert "results" in result
    assert "recommendation" in result
    assert len(result["results"]) > 0

def test_regression_pipeline():
    df = pd.DataFrame({
        "feat1": np.random.randn(100),
        "feat2": np.random.randn(100),
        "target": np.random.randn(100),
    })
    pipeline = MLPilotPipeline(artifact_folder="test_artifacts")
    result = pipeline.run(df, target_col="target", user_plan="trial", optimize=False)
    assert result["status"] == "completed"
    assert result["problem_type"] == "regression"

def test_unsupervised_pipeline():
    df = pd.DataFrame({
        "a": np.random.randn(50),
        "b": np.random.randn(50),
        "c": np.random.randn(50),
    })
    pipeline = MLPilotPipeline(artifact_folder="test_artifacts")
    result = pipeline.run(df, target_col=None, user_plan="trial", optimize=False)
    assert result["status"] == "completed"
    assert result["problem_type"] == "unsupervised"
    assert "clustering" in result