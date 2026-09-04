"""Experiment history and artifact tracking."""
from datetime import datetime
from extensions import db
import json

class Experiment(db.Model):
    __tablename__ = "experiments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    experiment_name = db.Column(db.String(200), nullable=False)
    dataset_name = db.Column(db.String(500), nullable=False)
    dataset_size_rows = db.Column(db.Integer, nullable=True)
    dataset_size_cols = db.Column(db.Integer, nullable=True)

    # Detection
    detected_problem = db.Column(db.String(100), nullable=True)
    detected_subtype = db.Column(db.String(100), nullable=True)

    # Preprocessing
    preprocessing_log = db.Column(db.Text, nullable=True)  # JSON
    features_used = db.Column(db.Text, nullable=True)  # JSON list
    target_column = db.Column(db.String(200), nullable=True)

    # Training
    models_tested = db.Column(db.Text, nullable=True)  # JSON
    best_model_name = db.Column(db.String(200), nullable=True)
    best_metric_name = db.Column(db.String(50), nullable=True)
    best_metric_value = db.Column(db.Float, nullable=True)

    # Optimization
    optimized = db.Column(db.Boolean, default=False)
    baseline_performance = db.Column(db.Float, nullable=True)
    optimized_performance = db.Column(db.Float, nullable=True)
    improvement_pct = db.Column(db.Float, nullable=True)

    # Diagnostics
    overfitting_status = db.Column(db.String(50), nullable=True)
    underfitting_status = db.Column(db.String(50), nullable=True)

    # Artifacts
    artifact_paths = db.Column(db.Text, nullable=True)  # JSON dict

    # Status
    status = db.Column(db.String(50), default="running")  # running, completed, failed
    error_message = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime, nullable=True)

    def set_preprocessing_log(self, log: list):
        self.preprocessing_log = json.dumps(log)

    def get_preprocessing_log(self) -> list:
        return json.loads(self.preprocessing_log or "[]")

    def set_models_tested(self, models: list):
        self.models_tested = json.dumps(models)

    def get_models_tested(self) -> list:
        return json.loads(self.models_tested or "[]")

    def set_features_used(self, features: list):
        self.features_used = json.dumps(features)

    def get_features_used(self) -> list:
        return json.loads(self.features_used or "[]")

    def set_artifact_paths(self, paths: dict):
        self.artifact_paths = json.dumps(paths)

    def get_artifact_paths(self) -> dict:
        return json.loads(self.artifact_paths or "{}")

    def __repr__(self):
        return f"<Experiment {self.experiment_name} ({self.status})>"