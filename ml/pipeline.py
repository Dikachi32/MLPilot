"""End-to-end AutoML pipeline orchestrator."""
import os
import uuid
import traceback
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml.detection import detect_problem_type, recommend_algorithms
from ml.preprocessing import PreprocessingEngine
from ml.feature_engineering import select_features_auto
from ml.evaluation import EvaluationEngine
from ml.optimization import OptimizationEngine
from ml.artifacts import ArtifactManager
from ml.models.classification import CLASSIFICATION_MODELS
from ml.models.regression import REGRESSION_MODELS
from ml.models.nlp import NLP_MODELS
from ml.models.unsupervised import UnsupervisedEngine
from ml.models.image_cnn import ImageDatasetLoader, CNNTrainer
from ml.models.ensemble import build_voting_ensemble

OPTIMIZED_SUFFIX = " (Optimized)"


def rank_results(results: List[Dict], problem_type: str) -> List[Dict]:
    """Rank result rows by primary_score. Rows without a score are dropped, never ranked."""
    scored = [r for r in results if r.get("primary_score") is not None]
    if problem_type in ("classification", "nlp", "image_classification"):
        scored.sort(key=lambda x: x["primary_score"], reverse=True)
    else:
        scored.sort(key=lambda x: x["primary_score"])
    return scored


class MLPilotPipeline:
    """Intelligent AutoML pipeline: Understand -> Prepare -> Select -> Train -> Evaluate -> Optimize -> Recommend."""

    def __init__(self, artifact_folder: str = "artifacts"):
        self.artifact_manager = ArtifactManager(artifact_folder)
        self.preproc = PreprocessingEngine()
        self.results: List[Dict] = []
        self.experiment_id = str(uuid.uuid4())[:8]
        self.metadata = {
            "experiment_id": self.experiment_id,
            "pipeline_version": "1.1.0",
            "steps": [],
        }

    def run(self, df: pd.DataFrame, target_col: Optional[str] = None,
            user_plan: str = "free", optimize: bool = True) -> Dict:
        """Execute the full AutoML pipeline."""
        try:
            self.metadata["dataset_shape"] = list(df.shape)
            self.metadata["target_column"] = target_col

            # Step 1: Detection
            detection = detect_problem_type(df, target_col)
            problem = detection["problem_type"]
            self.metadata["detected_problem"] = problem
            self.metadata["detection_confidence"] = detection.get("confidence")

            # Step 2: Route to the appropriate pipeline
            if problem == "image_classification":
                return self._run_image_pipeline(df, detection)
            elif problem == "unsupervised":
                return self._run_unsupervised_pipeline(df, detection)
            elif problem == "nlp":
                return self._run_nlp_pipeline(df, detection, target_col, optimize)
            else:
                return self._run_supervised_pipeline(df, detection, target_col, problem, optimize)

        except Exception as e:
            self.metadata["error"] = str(e)
            self.metadata["traceback"] = traceback.format_exc()
            return self._failed(str(e))

    def _failed(self, message: str) -> Dict:
        """Uniform failure payload so the template never meets a half-built result dict."""
        return {
            "status": "failed",
            "error": message,
            "experiment_id": self.experiment_id,
            "problem_type": self.metadata.get("detected_problem", "unknown"),
            "preprocessing": {},
            "results": [],
            "failed_models": [],
            "recommendation": None,
            "model_paths": {},
            "preproc_paths": {},
            "smote_applied": False,
            "metadata": self.metadata,
        }

    def _score_model(self, model, X_train, y_train, X_test, y_test, problem_type: str) -> Dict:
        """Evaluate an already-fitted model on train and test, and diagnose generalization."""
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        if problem_type == "classification":
            train_metrics = EvaluationEngine.evaluate_classification(y_train, train_pred)
            test_metrics = EvaluationEngine.evaluate_classification(y_test, test_pred)
            score = test_metrics["accuracy"]
            train_score = train_metrics["accuracy"]
            metric_type = "accuracy"
        else:
            train_metrics = EvaluationEngine.evaluate_regression(y_train, train_pred)
            test_metrics = EvaluationEngine.evaluate_regression(y_test, test_pred)
            score = test_metrics["rmse"]
            train_score = train_metrics["rmse"]
            metric_type = "rmse"

        diagnostics = EvaluationEngine.detect_overfitting_underfitting(
            train_score, score, metric_type
        )

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "primary_score": score,
            "train_score": train_score,
            "diagnostics": diagnostics,
        }

    def _run_supervised_pipeline(self, df: pd.DataFrame, detection: Dict,
                                 target_col: Optional[str], problem_type: str,
                                 optimize: bool) -> Dict:
        """Classification or regression pipeline."""
        if target_col is None or target_col not in df.columns:
            return self._failed("Target column required for supervised learning.")

        # Preprocess
        X, y, preproc_info = self.preproc.preprocess_tabular(
            df, target_col, problem_type, apply_pca=True, apply_smote=False
        )

        # Train/test split
        test_size = 0.2 if len(df) >= 50 else 0.3
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if problem_type == "classification" else None
        )

        # SMOTE on training data only
        X_train, y_train, smote_applied = self.preproc.apply_smote_safe(X_train, y_train, problem_type)

        # Select models
        model_names = recommend_algorithms(detection)
        registry = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
        models_to_run = {name: registry[name] for name in model_names if name in registry}

        # ---- Baseline training ----
        trained_models: Dict[str, Any] = {}
        baseline_results: List[Dict] = []
        failed_models: List[Dict] = []

        for name, model in models_to_run.items():
            try:
                model.fit(X_train, y_train)
                scored = self._score_model(model, X_train, y_train, X_test, y_test, problem_type)
                baseline_results.append({
                    "model": name,
                    "base_model": name,
                    "problem_type": problem_type,
                    "optimized": False,
                    **scored,
                })
                trained_models[name] = model
            except Exception as e:
                failed_models.append({"model": name, "error": str(e)})

        # ---- Optimization ----
        optimized_results: List[Dict] = []
        if optimize:
            for res in baseline_results:
                name = res["model"]
                opt_name = name + OPTIMIZED_SUFFIX
                try:
                    opt = OptimizationEngine.optimize(
                        trained_models[name], name, X_train, y_train, problem_type, n_iter=10
                    )
                except Exception as e:
                    failed_models.append({"model": opt_name, "error": str(e)})
                    continue

                if not opt.get("optimized"):
                    continue

                try:
                    scored = self._score_model(opt["model"], X_train, y_train, X_test, y_test, problem_type)
                except Exception as e:
                    failed_models.append({"model": opt_name, "error": str(e)})
                    continue

                baseline_score = res["primary_score"]
                opt_score = scored["primary_score"]

                # Higher is better for accuracy; lower is better for RMSE
                if problem_type == "classification":
                    improvement = opt_score - baseline_score
                else:
                    improvement = baseline_score - opt_score
                improvement_pct = round((improvement / abs(baseline_score)) * 100, 2) if baseline_score else 0.0

                optimized_results.append({
                    "model": opt_name,
                    "base_model": name,
                    "problem_type": problem_type,
                    "optimized": True,
                    "baseline_score": baseline_score,
                    "optimized_score": opt_score,
                    "improvement_pct": improvement_pct,
                    "best_params": opt.get("best_params", {}),
                    **scored,
                })
                trained_models[opt_name] = opt["model"]

        # ---- Ensemble ----
        # Use the best variant of each algorithm so one algorithm is not double-weighted.
        ensemble_result = None
        ensemble_members: Dict[str, Any] = {}
        for res in rank_results(baseline_results + optimized_results, problem_type):
            base = res.get("base_model", res["model"])
            if base not in ensemble_members:
                ensemble_members[base] = trained_models[res["model"]]

        if len(ensemble_members) >= 2:
            try:
                ensemble = build_voting_ensemble(ensemble_members, problem_type)
                if ensemble is not None:
                    ensemble.fit(X_train, y_train)
                    scored = self._score_model(ensemble, X_train, y_train, X_test, y_test, problem_type)
                    ensemble_result = {
                        "model": "Ensemble (Voting)",
                        "base_model": "Ensemble (Voting)",
                        "problem_type": problem_type,
                        "optimized": False,
                        **scored,
                    }
                    trained_models["Ensemble (Voting)"] = ensemble
            except Exception as e:
                failed_models.append({"model": "Ensemble (Voting)", "error": str(e)})

        # ---- Combine and rank ----
        all_results = baseline_results + optimized_results
        if ensemble_result:
            all_results.append(ensemble_result)
        all_results = rank_results(all_results, problem_type)

        # ---- Recommendation ----
        best = all_results[0] if all_results else None
        recommendation = None
        if best:
            recommendation = {
                "model": best["model"],
                "problem_type": problem_type,
                "metric": "Accuracy" if problem_type == "classification" else "RMSE",
                "score": best.get("primary_score"),
                "baseline_score": best.get("baseline_score"),
                "optimized_score": best.get("optimized_score"),
                "improvement_pct": best.get("improvement_pct"),
                "overfitting_status": (best.get("diagnostics") or {}).get("status"),
                "preprocessing_applied": self.preproc.cleaning_log,
                "smote_applied": smote_applied,
                "pca_applied": self.preproc.pca is not None,
            }

        # ---- Save artifacts ----
        model_paths = {}
        for name, mdl in trained_models.items():
            try:
                model_paths[name] = self.artifact_manager.save_model(mdl, name, self.experiment_id)
            except Exception as e:
                failed_models.append({"model": name, "error": f"Could not save artifact: {e}"})

        preproc_paths = self.preproc.save_artifacts(
            os.path.join(self.artifact_manager.base_folder, self.experiment_id), "preproc"
        )

        self.metadata["steps"] = ["detection", "preprocessing", "baseline_training",
                                  "optimization", "ensemble", "recommendation"]
        self.metadata["best_model"] = best["model"] if best else None
        self.metadata["feature_names"] = preproc_info.get("feature_names", [])

        return {
            "status": "completed",
            "experiment_id": self.experiment_id,
            "problem_type": problem_type,
            "detection": detection,
            "preprocessing": preproc_info,
            "smote_applied": smote_applied,
            "results": all_results,
            "failed_models": failed_models,
            "recommendation": recommendation,
            "model_paths": model_paths,
            "preproc_paths": preproc_paths,
            "metadata": self.metadata,
        }

    def _run_nlp_pipeline(self, df: pd.DataFrame, detection: Dict,
                          target_col: Optional[str], optimize: bool) -> Dict:
        """NLP text classification pipeline."""
        text_cols = detection["analysis"]["text_columns"]
        if not text_cols:
            return self._failed("No text columns found for NLP.")

        text_col = text_cols[0]
        y = df[target_col].copy() if target_col and target_col in df.columns else None
        if y is None:
            return self._failed("Target column required for NLP classification.")

        if y.dtype == "object":
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            self.preproc.target_encoder = le

        X = self.preproc.preprocess_text(df[text_col], fit=True, max_features=5000)

        test_size = 0.2 if len(df) >= 50 else 0.3
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        results: List[Dict] = []
        failed_models: List[Dict] = []
        trained: Dict[str, Any] = {}

        for name, model in NLP_MODELS.items():
            try:
                model.fit(X_train, y_train)
                train_metrics = EvaluationEngine.evaluate_classification(y_train, model.predict(X_train))
                test_metrics = EvaluationEngine.evaluate_classification(y_test, model.predict(X_test))
                diagnostics = EvaluationEngine.detect_overfitting_underfitting(
                    train_metrics["accuracy"], test_metrics["accuracy"], "accuracy"
                )
                results.append({
                    "model": name,
                    "base_model": name,
                    "problem_type": "nlp",
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "primary_score": test_metrics["accuracy"],
                    "train_score": train_metrics["accuracy"],
                    "diagnostics": diagnostics,
                    "optimized": False,
                })
                trained[name] = model
            except Exception as e:
                failed_models.append({"model": name, "error": str(e)})

        results = rank_results(results, "nlp")
        best = results[0] if results else None

        model_paths = {}
        for name, m in trained.items():
            try:
                model_paths[name] = self.artifact_manager.save_model(m, name, self.experiment_id)
            except Exception as e:
                failed_models.append({"model": name, "error": f"Could not save artifact: {e}"})

        preproc_paths = self.preproc.save_artifacts(
            os.path.join(self.artifact_manager.base_folder, self.experiment_id), "preproc"
        )

        self.metadata["best_model"] = best["model"] if best else None

        return {
            "status": "completed",
            "experiment_id": self.experiment_id,
            "problem_type": "nlp",
            "detection": detection,
            "preprocessing": {"cleaning_log": self.preproc.cleaning_log, "feature_names": []},
            "smote_applied": False,
            "results": results,
            "failed_models": failed_models,
            "recommendation": {
                "model": best["model"],
                "problem_type": "nlp",
                "metric": "Accuracy",
                "score": best["primary_score"],
                "overfitting_status": (best.get("diagnostics") or {}).get("status"),
                "preprocessing_applied": self.preproc.cleaning_log,
            } if best else None,
            "model_paths": model_paths,
            "preproc_paths": preproc_paths,
            "metadata": self.metadata,
        }

    def _run_unsupervised_pipeline(self, df: pd.DataFrame, detection: Dict) -> Dict:
        """Unsupervised learning pipeline (clustering + PCA)."""
        df_num = df.select_dtypes(include=[np.number]).dropna()
        if df_num.empty:
            return self._failed("No numeric columns available for unsupervised learning.")

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(df_num)

        cluster_results = UnsupervisedEngine.auto_cluster(X, max_k=min(10, len(X) - 1))
        if not cluster_results:
            return self._failed("Not enough rows to form clusters.")

        best_k = max(cluster_results, key=lambda x: x.get("silhouette") or -1)
        pca_result = UnsupervisedEngine.run_pca(X, n_components=0.95)
        hier_result = UnsupervisedEngine.run_hierarchical(X, n_clusters=best_k["k"])

        return {
            "status": "completed",
            "experiment_id": self.experiment_id,
            "problem_type": "unsupervised",
            "detection": detection,
            "preprocessing": {"cleaning_log": [], "feature_names": list(df_num.columns)},
            "smote_applied": False,
            "results": [],
            "failed_models": [],
            "recommendation": {
                "model": "K-Means (k=%s)" % best_k["k"],
                "problem_type": "unsupervised",
                "metric": "Silhouette",
                "score": best_k.get("silhouette"),
            },
            "clustering": {
                "best_k": best_k["k"],
                "best_silhouette": best_k.get("silhouette"),
                "all_k_results": [
                    {"k": r["k"], "silhouette": r.get("silhouette"), "inertia": r.get("inertia")}
                    for r in cluster_results
                ],
            },
            "hierarchical": {
                "n_clusters": hier_result["n_clusters"],
                "silhouette": hier_result.get("silhouette"),
            },
            "pca": {
                "n_components": pca_result["n_components"],
                "total_variance": pca_result["total_variance"],
            },
            "model_paths": {},
            "preproc_paths": {},
            "metadata": self.metadata,
        }

    def _run_image_pipeline(self, df: pd.DataFrame, detection: Dict) -> Dict:
        """Image classification CNN pipeline."""
        img_cols = detection["analysis"]["image_columns"]
        if not img_cols:
            return self._failed("No image columns detected.")

        img_col = img_cols[0]
        label_col = None
        for c in df.columns:
            if c != img_col and df[c].dtype == "object":
                label_col = c
                break

        if label_col is None:
            return self._failed("No label column found for image classification.")

        loader = ImageDatasetLoader(img_size=(128, 128))
        try:
            X, y = loader.load_from_paths(df[img_col].tolist(), df[label_col].tolist())
        except Exception as e:
            return self._failed("Image loading failed: %s" % e)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        num_classes = len(np.unique(y))
        try:
            model = CNNTrainer.build_model(X_train.shape[1:], num_classes)
        except ImportError as e:
            return self._failed(str(e))

        train_result = CNNTrainer.train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
        if "error" in train_result:
            return self._failed(train_result["error"])

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        test_acc = round(test_acc * 100, 2)

        model_path = self.artifact_manager.save_model(model, "CNN", self.experiment_id)
        encoder_path = os.path.join(self.artifact_manager.base_folder, self.experiment_id, "image_label_encoder.pkl")
        loader.save_encoder(encoder_path)

        diagnostics = EvaluationEngine.detect_overfitting_underfitting(
            train_result["final_train_acc"], test_acc, "accuracy"
        )

        return {
            "status": "completed",
            "experiment_id": self.experiment_id,
            "problem_type": "image_classification",
            "detection": detection,
            "preprocessing": {"cleaning_log": [], "feature_names": []},
            "smote_applied": False,
            "results": [{
                "model": "CNN",
                "base_model": "CNN",
                "problem_type": "image_classification",
                "train_metrics": {"accuracy": train_result["final_train_acc"]},
                "test_metrics": {"accuracy": test_acc},
                "primary_score": test_acc,
                "train_score": train_result["final_train_acc"],
                "diagnostics": diagnostics,
                "optimized": False,
            }],
            "failed_models": [],
            "recommendation": {
                "model": "CNN",
                "problem_type": "image_classification",
                "metric": "Accuracy",
                "score": test_acc,
                "overfitting_status": diagnostics.get("status"),
            },
            "train_accuracy": train_result["final_train_acc"],
            "val_accuracy": train_result["final_val_acc"],
            "test_accuracy": test_acc,
            "model_paths": {"CNN": model_path, "label_encoder": encoder_path},
            "preproc_paths": {},
            "metadata": self.metadata,
        }