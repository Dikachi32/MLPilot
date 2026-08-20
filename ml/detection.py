"""Intelligent dataset understanding and problem-type detection."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter

def infer_column_type(series: pd.Series) -> str:
    """Infer semantic type of a column."""
    if series.dtype == "object":
        # Try numeric
        converted = pd.to_numeric(series, errors="coerce")
        if converted.notna().sum() / len(series) > 0.8:
            return "numeric"
        # Try datetime
        try:
            dt = pd.to_datetime(series, errors="coerce")
            if dt.notna().sum() / len(series) > 0.8:
                return "datetime"
        except Exception:
            pass
        # Try image path
        sample = series.dropna().astype(str).head(20)
        image_exts = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff")
        if any(str(v).lower().endswith(image_exts) for v in sample):
            return "image_path"
        return "text"
    elif np.issubdtype(series.dtype, np.number):
        return "numeric"
    elif np.issubdtype(series.dtype, np.datetime64):
        return "datetime"
    return "categorical"


def analyze_dataset(df: pd.DataFrame) -> Dict:
    """Comprehensive dataset profiling."""
    n_rows, n_cols = df.shape
    analysis = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "columns": {},
        "missing_values": {},
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": [],
        "high_cardinality_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
        "text_columns": [],
        "datetime_columns": [],
        "image_columns": [],
    }

    for col in df.columns:
        series = df[col]
        col_type = infer_column_type(series)
        unique_count = series.nunique(dropna=True)
        missing_count = int(series.isna().sum())
        missing_pct = round(missing_count / n_rows * 100, 2) if n_rows > 0 else 0

        analysis["columns"][col] = {
            "type": col_type,
            "dtype": str(series.dtype),
            "unique": unique_count,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
        analysis["missing_values"][col] = missing_pct

        if col_type == "numeric":
            analysis["numeric_columns"].append(col)
        elif col_type == "text":
            analysis["text_columns"].append(col)
        elif col_type == "datetime":
            analysis["datetime_columns"].append(col)
        elif col_type == "image_path":
            analysis["image_columns"].append(col)
        else:
            analysis["categorical_columns"].append(col)

        if unique_count == 1:
            analysis["constant_columns"].append(col)
        if col_type in ("categorical", "text") and unique_count > n_rows * 0.5:
            analysis["high_cardinality_columns"].append(col)

    return analysis


def detect_problem_type(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict:
    """Detect ML problem type from dataset."""
    analysis = analyze_dataset(df)
    n_rows = analysis["n_rows"]

    # Image dataset detection
    if analysis["image_columns"]:
        return {
            "problem_type": "image_classification",
            "subtype": "cnn",
            "confidence": "high",
            "analysis": analysis,
            "message": "Image dataset detected. A CNN pipeline will be used.",
        }

    # NLP detection: single text column, small row count relative to text
    if len(analysis["text_columns"]) == 1 and len(analysis["numeric_columns"]) == 0 and len(analysis["categorical_columns"]) <= 1:
        text_col = analysis["text_columns"][0]
        avg_len = df[text_col].astype(str).str.len().mean()
        if avg_len > 20:
            return {
                "problem_type": "nlp",
                "subtype": "text_classification",
                "confidence": "high",
                "analysis": analysis,
                "message": "Text/NLP dataset detected. TF-IDF vectorization will be applied.",
            }

    # If no target provided, suggest unsupervised
    if target_col is None or target_col not in df.columns:
        return {
            "problem_type": "unsupervised",
            "subtype": "clustering_or_pca",
            "confidence": "medium",
            "analysis": analysis,
            "message": "No target column specified. Unsupervised learning (clustering / PCA) is recommended.",
        }

    y = df[target_col]
    y_unique = y.nunique(dropna=True)

    # Classification vs Regression
    if y_unique <= 10 or (y.dtype == "object" and y_unique <= 100):
        # Check class balance
        class_counts = y.value_counts(normalize=True)
        min_ratio = class_counts.min() / class_counts.max() if len(class_counts) > 1 else 1.0
        imbalance = min_ratio < 0.3

        return {
            "problem_type": "classification",
            "subtype": "multiclass" if y_unique > 2 else "binary",
            "confidence": "high",
            "analysis": analysis,
            "class_distribution": class_counts.to_dict(),
            "imbalanced": imbalance,
            "message": f"Classification detected ({y_unique} classes). {'Class imbalance detected.' if imbalance else 'Classes appear balanced.'}",
        }
    else:
        return {
            "problem_type": "regression",
            "subtype": "continuous",
            "confidence": "high",
            "analysis": analysis,
            "message": "Regression detected (continuous target).",
        }


def suggest_target_columns(df: pd.DataFrame, analysis: Dict) -> List[str]:
    """Suggest potential target columns."""
    candidates = []
    target_names = ["target", "label", "class", "y", "outcome", "result", "prediction",
                    "predict", "category", "type", "status", "grade", "score", "rating",
                    "rank", "decision", "flag", "spam", "ham", "sentiment", "class_label"]

    for col in df.columns:
        col_lower = col.lower()
        if any(t in col_lower for t in target_names):
            unique = df[col].nunique(dropna=True)
            if 2 <= unique <= 100:
                candidates.append(col)

    # Fallback: last few columns with reasonable cardinality
    if not candidates:
        for col in reversed(df.columns.tolist()[-3:]):
            unique = df[col].nunique(dropna=True)
            if 2 <= unique <= 100:
                candidates.append(col)
                break

    return candidates


def recommend_algorithms(detection_result: Dict) -> List[str]:
    """Recommend candidate algorithms based on detection."""
    pt = detection_result.get("problem_type", "unknown")

    if pt == "classification":
        return [
            "Logistic Regression", "K-Nearest Neighbors", "Decision Tree",
            "Random Forest", "Support Vector Machine", "Gradient Boosting",
            "AdaBoost", "Extra Trees", "Bagging"
        ]
    elif pt == "regression":
        return [
            "Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net",
            "Decision Tree Regressor", "Random Forest Regressor", "KNN Regressor",
            "Support Vector Regressor", "Gradient Boosting Regressor",
            "AdaBoost Regressor", "Extra Trees Regressor", "Bagging Regressor"
        ]
    elif pt == "unsupervised":
        return ["K-Means", "Hierarchical Clustering", "PCA"]
    elif pt == "image_classification":
        return ["CNN"]
    elif pt == "nlp":
        return ["Logistic Regression", "Random Forest", "Gradient Boosting", "SVM"]
    return []