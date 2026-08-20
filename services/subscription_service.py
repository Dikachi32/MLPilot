"""Subscription entitlement and plan management."""
from datetime import datetime, timedelta
from typing import Optional
from models.user import User

class SubscriptionService:
    def __init__(self, plan_config: dict):
        self.plan_config = plan_config

    def get_plan(self, user: User) -> dict:
        """Get plan config for user, fallback to free."""
        plan = self.plan_config.get(user.plan)
        if plan is None:
            plan = self.plan_config.get("free", {})
        return plan

    def can_use_feature(self, user: User, feature: str) -> bool:
        plan = self.get_plan(user)
        if feature == "cnn":
            return plan.get("allow_cnn", False)
        if feature == "advanced_optimization":
            return plan.get("allow_advanced_optimization", False)
        if feature == "unsupervised":
            return plan.get("allow_unsupervised", False)
        return True

    def remaining_quota(self, user: User) -> dict:
        plan = self.get_plan(user)
        return {
            "datasets": max(0, plan.get("max_datasets", 0) - user.datasets_uploaded),
            "training_runs": max(0, plan.get("max_training_runs", 0) - user.training_runs_used),
            "optimization_runs": max(0, plan.get("max_optimization_runs", 0) - user.optimization_runs_used),
            "downloads": max(0, plan.get("max_downloads", 0) - user.downloads_used),
            "max_dataset_size_mb": plan.get("max_dataset_size_mb", 10),
        }

    def start_trial(self, user: User):
        user.plan = "trial"
        user.trial_started_at = datetime.utcnow()
        user.subscription_status = "active"
        from extensions import db
        db.session.commit()

    def activate_plan(self, user: User, plan: str, duration_days: int = 30):
        user.plan = plan
        user.subscription_status = "active"
        user.subscription_expires_at = datetime.utcnow() + timedelta(days=duration_days)
        from extensions import db
        db.session.commit()

    def check_expiry(self, user: User) -> bool:
        if user.plan in ("free",):
            return True
        if user.plan == "trial" and not user.is_trial_active():
            user.subscription_status = "expired"
            from extensions import db
            db.session.commit()
            return False
        if user.subscription_expires_at and datetime.utcnow() > user.subscription_expires_at:
            user.subscription_status = "expired"
            from extensions import db
            db.session.commit()
            return False
        return True