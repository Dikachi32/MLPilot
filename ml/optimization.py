"""Hyperparameter optimization engine."""
import numpy as np
from typing import Dict, Any, Optional, Callable
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
import warnings

class OptimizationEngine:
    """Optimize model hyperparameters using RandomizedSearchCV."""

    PARAM_GRIDS = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
            "max_iter": [1000],
        },
        "K-Nearest Neighbors": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5],
        },
        "Support Vector Machine": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto", 0.01, 0.001],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "AdaBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.5, 1.0, 1.5],
        },
        "Extra Trees": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
        },
        "Bagging": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.7, 1.0],
        },
        "Linear Regression": {},
        "Ridge Regression": {"alpha": [0.1, 1.0, 10.0, 100.0]},
        "Lasso Regression": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "Elastic Net": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]},
        "Decision Tree Regressor": {
            "max_depth": [3, 5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
        },
        "KNN Regressor": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"],
        },
        "Support Vector Regressor": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "epsilon": [0.01, 0.1, 0.5],
        },
        "Gradient Boosting Regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "AdaBoost Regressor": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.5, 1.0, 1.5],
        },
        "Extra Trees Regressor": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, 20, None],
        },
        "Bagging Regressor": {
            "n_estimators": [10, 50, 100],
            "max_samples": [0.5, 0.7, 1.0],
        },
    }

    @classmethod
    def optimize(cls, model, model_name: str, X_train: Any, y_train: Any,
                 problem_type: str, n_iter: int = 10, cv: int = 3) -> Dict:
        """Run RandomizedSearchCV and return best model + metrics."""
        grid = cls.PARAM_GRIDS.get(model_name, {})
        if not grid:
            return {"model": model, "best_params": {}, "best_score": None, "optimized": False}

        scoring = "accuracy" if problem_type == "classification" else "neg_mean_squared_error"

        try:
            search = RandomizedSearchCV(
                model, grid, n_iter=min(n_iter, 20), cv=min(cv, 3),
                scoring=scoring, random_state=42, n_jobs=-1, error_score="raise"
            )
            search.fit(X_train, y_train)
            return {
                "model": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": round(search.best_score_ * 100, 2) if problem_type == "classification" else round(-search.best_score_, 4),
                "optimized": True,
            }
        except Exception as e:
            return {"model": model, "best_params": {}, "best_score": None, "optimized": False, "error": str(e)}