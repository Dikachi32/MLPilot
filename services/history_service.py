"""Experiment history persistence."""
from models.experiment import Experiment, db
from typing import List, Optional

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
        return Experiment.query.filter_by(user_id=user_id).order_by(Experiment.created_at.desc()).limit(limit).all()

    @staticmethod
    def get_experiment(exp_id: int, user_id: int) -> Optional[Experiment]:
        return Experiment.query.filter_by(id=exp_id, user_id=user_id).first()

    @staticmethod
    def update_experiment(exp: Experiment, **kwargs):
        for k, v in kwargs.items():
            if hasattr(exp, k):
                setattr(exp, k, v)
        db.session.commit()

    @staticmethod
    def finalize_experiment(exp: Experiment, status: str = "completed", error: str = None):
        from datetime import datetime
        exp.status = status
        if error:
            exp.error_message = error
        exp.completed_at = datetime.utcnow()
        db.session.commit()