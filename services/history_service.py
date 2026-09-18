"""Experiment history persistence with full JSON state support."""
import json
from datetime import datetime
from typing import List, Optional

from models.experiment import Experiment, db

# Columns that hold JSON text but may be handed a tuple/list/dict by the caller.
_JSON_TEXT_FIELDS = {
    "dataset_shape", "result_json", "preprocessing_log",
    "models_tested", "artifact_paths", "features_used",
}


def _coerce(field: str, value):
    """SQLite cannot bind a tuple to a String column — serialise those first."""
    if field in _JSON_TEXT_FIELDS and value is not None and not isinstance(value, str):
        return json.dumps(value, default=str)
    return value


class HistoryService:
    @staticmethod
    def create_experiment(user_id: int, name: str, dataset_name: str) -> Experiment:
        exp = Experiment(
            user_id=user_id,
            experiment_name=name,
            dataset_name=dataset_name,
            status="running"
        )
        db.session.add(exp)
        db.session.commit()
        return exp

    @staticmethod
    def get_user_experiments(user_id: int, limit: int = 50) -> List[Experiment]:
        return (Experiment.query
                .filter_by(user_id=user_id)
                .order_by(Experiment.created_at.desc())
                .limit(limit).all())

    @staticmethod
    def get_experiment(exp_id: int, user_id: int) -> Optional[Experiment]:
        return Experiment.query.filter_by(id=exp_id, user_id=user_id).first()

    @staticmethod
    def update_experiment(exp: Experiment, **kwargs):
        for k, v in kwargs.items():
            if hasattr(exp, k):
                setattr(exp, k, _coerce(k, v))
        db.session.commit()

    @staticmethod
    def finalize_experiment(exp: Experiment, status: str = "completed", error: str = None):
        exp.status = status
        if error:
            exp.error_message = error
        exp.completed_at = datetime.utcnow()
        db.session.commit()

    # -- JSON field setters for full state persistence --
    @staticmethod
    def set_preprocessing_log(exp: Experiment, log_json: str):
        exp.preprocessing_log = _coerce("preprocessing_log", log_json)
        db.session.commit()

    @staticmethod
    def set_models_tested(exp: Experiment, models_json: str):
        exp.models_tested = _coerce("models_tested", models_json)
        db.session.commit()

    @staticmethod
    def set_artifact_paths(exp: Experiment, paths_json: str):
        exp.artifact_paths = _coerce("artifact_paths", paths_json)
        db.session.commit()

    @staticmethod
    def set_features_used(exp: Experiment, features_json: str):
        exp.features_used = _coerce("features_used", features_json)
        db.session.commit()

    @staticmethod
    def set_result_json(exp: Experiment, result_json: str):
        exp.result_json = _coerce("result_json", result_json)
        db.session.commit()