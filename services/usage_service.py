"""Usage tracking and limit enforcement."""
from flask import current_app
from models.user import User

class UsageService:
    def __init__(self, user: User):
        self.user = user

    def check_and_increment(self, action: str) -> tuple:
        cfg = self.user.get_plan_config(current_app.config)

        if action == "dataset_upload":
            if self.user.datasets_uploaded >= cfg["max_datasets"]:
                return False, f"Dataset upload limit reached ({cfg['max_datasets']}). Upgrade to continue."
            self.user.increment_usage("datasets_uploaded")
            return True, "OK"

        if action == "train":
            if self.user.training_runs_used >= cfg["max_training_runs"]:
                return False, f"Training run limit reached ({cfg['max_training_runs']}). Upgrade to continue."
            self.user.increment_usage("training_runs_used")
            return True, "OK"

        if action == "optimize":
            if not cfg.get("allow_advanced_optimization", False):
                return False, "Optimization is not available on your current plan. Upgrade to Pro."
            if self.user.optimization_runs_used >= cfg["max_optimization_runs"]:
                return False, f"Optimization limit reached ({cfg['max_optimization_runs']}). Upgrade to continue."
            self.user.increment_usage("optimization_runs_used")
            return True, "OK"

        if action == "download":
            if self.user.downloads_used >= cfg["max_downloads"]:
                return False, f"Download limit reached ({cfg['max_downloads']}). Upgrade to continue."
            self.user.increment_usage("downloads_used")
            return True, "OK"

        return True, "OK"