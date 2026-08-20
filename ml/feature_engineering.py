"""Intelligent feature selection and engineering."""
import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, SelectKBest

def select_features_auto(X: pd.DataFrame, y: pd.Series, problem_type: str,
                         max_features: int = 200) -> List[str]:
    """Select top features using mutual information."""
    if X.shape[1] <= max_features:
        return list(X.columns)

    # Ensure numeric
    X_num = X.select_dtypes(include=[np.number]).fillna(0)
    if X_num.shape[1] == 0:
        return list(X.columns)[:max_features]

    if problem_type == "classification":
        mi = mutual_info_classif(X_num, y, random_state=42)
    else:
        mi = mutual_info_regression(X_num, y, random_state=42)

    scores = pd.Series(mi, index=X_num.columns)
    top = scores.nlargest(max_features).index.tolist()
    return top


def engineer_datetime_features(df: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
    """Decompose datetime columns into useful numeric features."""
    df = df.copy()
    for col in datetime_cols:
        if col not in df.columns:
            continue
        dt = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = dt.dt.year.fillna(dt.dt.year.median())
        df[f"{col}_month"] = dt.dt.month.fillna(dt.dt.month.median())
        df[f"{col}_day"] = dt.dt.day.fillna(dt.dt.day.median())
        df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0)
        df = df.drop(columns=[col])
    return df