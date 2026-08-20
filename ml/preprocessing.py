"""Centralized preprocessing engine for tabular, text, and image data."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import scipy.stats as stats
import joblib
import os

def process_text(text: str) -> str:
    """Normalize text: lowercase, strip whitespace, handle None."""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


class PreprocessingEngine:
    """End-to-end preprocessing with artifact persistence."""

    def __init__(self):
        self.scaler: Optional[Any] = None
        self.pca: Optional[Any] = None
        self.vectorizer: Optional[Any] = None
        self.label_encoders: Dict[str, Any] = {}
        self.target_encoder: Optional[Any] = None
        self.numeric_impute_values: Dict[str, float] = {}
        self.dropped_columns: List[Dict] = []
        self.cleaning_log: List[str] = []
        self.feature_names: List[str] = []
        self.text_pipeline_fitted = False

    def analyze_and_drop_useless(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Intelligently remove useless columns and log reasons."""
        df = df.copy()
        n_rows = len(df)

        for col in df.columns:
            if col == target_col:
                continue

            series = df[col]
            unique = series.nunique(dropna=True)
            missing_pct = series.isna().sum() / n_rows * 100

            # Empty or constant
            if unique <= 1:
                self.dropped_columns.append({"column": col, "reason": "Constant or single unique value"})
                df = df.drop(columns=[col])
                continue

            # Excessive missing
            if missing_pct > 80:
                self.dropped_columns.append({"column": col, "reason": f"Excessive missing values ({missing_pct:.1f}%)"})
                df = df.drop(columns=[col])
                continue

            # ID-like columns
            if unique == n_rows and ("id" in col.lower() or "index" in col.lower() or col.lower() in ("id", "idx", "index", "key")):
                self.dropped_columns.append({"column": col, "reason": "Likely ID/index column (all unique)"})
                df = df.drop(columns=[col])
                continue

            # High cardinality text/categorical
            if series.dtype == "object" and unique > min(500, n_rows * 0.5):
                self.dropped_columns.append({"column": col, "reason": f"High cardinality ({unique} unique values)"})
                df = df.drop(columns=[col])
                continue

        self.cleaning_log.append(f"Analyzed {len(df.columns)} columns; dropped {len(self.dropped_columns)} useless/constant/ID columns.")
        return df

    def preprocess_tabular(self, df: pd.DataFrame, target_col: Optional[str] = None,
                           problem_type: str = "classification", apply_pca: bool = True,
                           apply_smote: bool = True) -> Tuple[Any, Any, Dict]:
        """Preprocess tabular data: clean, encode, scale, optional PCA/SMOTE."""
        df = self.analyze_and_drop_useless(df, target_col)

        X = df.drop(columns=[target_col]) if target_col and target_col in df.columns else df.copy()
        y = df[target_col].copy() if target_col and target_col in df.columns else None

        # Separate types
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

        # Numeric processing
        for col in numeric_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median_val = X[col].median()
            self.numeric_impute_values[col] = median_val
            X[col] = X[col].fillna(median_val)

        # Outlier handling for regression
        if problem_type == "regression" and numeric_cols:
            z = np.abs(stats.zscore(X[numeric_cols].fillna(0)))
            mask = (z < 3).all(axis=1)
            removed = (~mask).sum()
            if removed > 0:
                X = X[mask]
                if y is not None:
                    y = y[mask]
                self.cleaning_log.append(f"Removed {removed} outlier rows (Z-score > 3).")

        # Categorical processing
        for col in categorical_cols:
            unique = X[col].nunique(dropna=True)
            if unique <= 2:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna("Missing"))
                self.label_encoders[col] = le
                self.cleaning_log.append(f"'{col}': Binary label encoded ({unique} categories).")
            elif unique <= 50:
                dummies = pd.get_dummies(X[col].astype(str).fillna("Missing"), prefix=col, drop_first=True)
                X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                self.cleaning_log.append(f"'{col}': One-hot encoded ({unique} categories).")
            else:
                freq = X[col].value_counts().to_dict()
                X[col] = X[col].map(freq).fillna(0)
                self.cleaning_log.append(f"'{col}': Frequency encoded ({unique} unique values).")

        # Target processing
        if y is not None and y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            self.target_encoder = le
            self.cleaning_log.append("Target converted from text to numeric labels.")

        # Drop rows with missing target
        if y is not None and y.isna().any():
            valid = y.notna()
            X = X[valid]
            y = y[valid]
            self.cleaning_log.append(f"Removed {(~valid).sum()} rows with missing target.")

        # Ensure all numeric
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        # Remove constant columns after encoding
        const_cols = [c for c in X.columns if X[c].nunique() <= 1]
        if const_cols:
            X = X.drop(columns=const_cols)
            self.cleaning_log.append(f"Removed {len(const_cols)} constant columns after encoding.")

        # Scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = list(X.columns)
        self.cleaning_log.append(f"Applied StandardScaler to {len(self.feature_names)} features.")

        # PCA
        if apply_pca and X_scaled.shape[1] > 50:
            n_comp = min(50, X_scaled.shape[1], len(X_scaled) - 1)
            self.pca = PCA(n_components=n_comp)
            X_final = self.pca.fit_transform(X_scaled)
            var = sum(self.pca.explained_variance_ratio_) * 100
            self.cleaning_log.append(f"PCA: {X_scaled.shape[1]} -> {n_comp} components ({var:.1f}% variance retained).")
        else:
            X_final = X_scaled

        # SMOTE (only on training data later; this returns raw for pipeline)
        return X_final, y, {
            "dropped_columns": self.dropped_columns,
            "cleaning_log": self.cleaning_log,
            "feature_names": self.feature_names,
            "n_features_final": X_final.shape[1],
        }

    def preprocess_text(self, texts: pd.Series, fit: bool = True, max_features: int = 5000) -> Any:
        """Fit or transform text using TF-IDF."""
        processed = texts.fillna("").astype(str).apply(process_text)
        if fit:
            self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
            X = self.vectorizer.fit_transform(processed)
            self.text_pipeline_fitted = True
        else:
            if self.vectorizer is None:
                raise ValueError("Text vectorizer not fitted yet.")
            X = self.vectorizer.transform(processed)
        return X

    def apply_smote_safe(self, X_train: Any, y_train: Any, problem_type: str) -> Tuple[Any, Any, bool]:
        """Apply SMOTE only to training data when appropriate."""
        if problem_type != "classification":
            return X_train, y_train, False
        counts = pd.Series(y_train).value_counts()
        if len(counts) < 2:
            return X_train, y_train, False
        min_ratio = counts.min() / counts.max()
        if min_ratio >= 0.3:
            return X_train, y_train, False
        min_class = counts.min()
        if min_class < 6:
            return X_train, y_train, False
        try:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_class - 1))
            X_res, y_res = smote.fit_resample(X_train, y_train)
            self.cleaning_log.append("SMOTE applied to training data to address class imbalance.")
            return X_res, y_res, True
        except Exception as e:
            self.cleaning_log.append(f"SMOTE skipped: {str(e)}")
            return X_train, y_train, False

    def save_artifacts(self, folder: str, prefix: str = "preproc") -> Dict[str, str]:
        """Save all preprocessing artifacts."""
        os.makedirs(folder, exist_ok=True)
        paths = {}
        if self.scaler:
            p = os.path.join(folder, f"{prefix}_scaler.pkl")
            joblib.dump(self.scaler, p)
            paths["scaler"] = p
        if self.pca:
            p = os.path.join(folder, f"{prefix}_pca.pkl")
            joblib.dump(self.pca, p)
            paths["pca"] = p
        if self.vectorizer:
            p = os.path.join(folder, f"{prefix}_vectorizer.pkl")
            joblib.dump(self.vectorizer, p)
            paths["vectorizer"] = p
        if self.target_encoder:
            p = os.path.join(folder, f"{prefix}_target_encoder.pkl")
            joblib.dump(self.target_encoder, p)
            paths["target_encoder"] = p
        for col, enc in self.label_encoders.items():
            safe_col = col.replace("/", "_").replace("\\", "_")
            p = os.path.join(folder, f"{prefix}_le_{safe_col}.pkl")
            joblib.dump(enc, p)
            paths[f"le_{col}"] = p
        return paths