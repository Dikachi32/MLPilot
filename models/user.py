"""User model with authentication and subscription state."""
from datetime import datetime, timedelta
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from extensions import db
import uuid

class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Subscription
    plan = db.Column(db.String(50), default="free")  # free, trial, pro_weekly, pro_monthly
    subscription_status = db.Column(db.String(50), default="active")  # active, expired, cancelled
    subscription_expires_at = db.Column(db.DateTime, nullable=True)
    trial_started_at = db.Column(db.DateTime, nullable=True)

    # Usage counters
    datasets_uploaded = db.Column(db.Integer, default=0)
    training_runs_used = db.Column(db.Integer, default=0)
    optimization_runs_used = db.Column(db.Integer, default=0)
    downloads_used = db.Column(db.Integer, default=0)

    # Relations
    experiments = db.relationship("Experiment", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)

    def is_trial_active(self) -> bool:
        if self.plan != "trial" or not self.trial_started_at:
            return False
        return datetime.utcnow() < self.trial_started_at + timedelta(days=14)

    def is_subscription_active(self) -> bool:
        if self.plan in ("free", "trial"):
            return self.plan != "trial" or self.is_trial_active()
        if self.subscription_status != "active":
            return False
        if self.subscription_expires_at:
            return datetime.utcnow() < self.subscription_expires_at
        return True

    def get_plan_config(self, app_config: dict) -> dict:
        # Handle both full app.config (has PLANS key) and raw plans dict
        if "PLANS" in app_config:
            plans = app_config.get("PLANS", {})
        else:
            plans = app_config
        return plans.get(self.plan, plans.get("free", {}))

    def can_upload_dataset(self, app_config: dict) -> bool:
        cfg = self.get_plan_config(app_config)
        return self.datasets_uploaded < cfg["max_datasets"]

    def can_train(self, app_config: dict) -> bool:
        cfg = self.get_plan_config(app_config)
        return self.training_runs_used < cfg["max_training_runs"]

    def can_optimize(self, app_config: dict) -> bool:
        cfg = self.get_plan_config(app_config)
        return self.optimization_runs_used < cfg["max_optimization_runs"]

    def can_download(self, app_config: dict) -> bool:
        cfg = self.get_plan_config(app_config)
        return self.downloads_used < cfg["max_downloads"]

    def increment_usage(self, field: str):
        setattr(self, field, getattr(self, field, 0) + 1)
        db.session.commit()

    def __repr__(self):
        return f"<User {self.username} ({self.plan})>"