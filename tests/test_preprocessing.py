"""Tests for preprocessing engine."""
import pytest
import pandas as pd
import numpy as np
from ml.preprocessing import PreprocessingEngine, process_text

def test_process_text():
    assert process_text("  Hello WORLD  ") == "hello world"
    assert process_text(None) == ""
    assert process_text(123) == "123"

def test_drop_useless_columns():
    engine = PreprocessingEngine()
    df = pd.DataFrame({
        "good": [1, 2, 3, 4, 5],
        "constant": [1, 1, 1, 1, 1],
        "mostly_missing": [1, np.nan, np.nan, np.nan, np.nan],
        "id_col": [1, 2, 3, 4, 5],
    })
    cleaned = engine.analyze_and_drop_useless(df)
    assert "constant" not in cleaned.columns
    assert "mostly_missing" not in cleaned.columns
    assert "id_col" not in cleaned.columns
    assert "good" in cleaned.columns
    assert len(engine.dropped_columns) == 3

def test_preprocess_tabular():
    engine = PreprocessingEngine()
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, 5],
        "cat": ["a", "b", "a", "b", "a"],
        "target": [0, 1, 0, 1, 0],
    })
    X, y, info = engine.preprocess_tabular(df, target_col="target", problem_type="classification")
    assert X is not None
    assert y is not None
    assert len(info["cleaning_log"]) > 0

def test_smote_safe():
    engine = PreprocessingEngine()
    X = np.random.randn(20, 5)
    y = np.array([0]*17 + [1]*3)
    X_res, y_res, applied = engine.apply_smote_safe(X, y, "classification")
    assert applied == True
    assert len(np.unique(y_res)) == 2