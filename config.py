"""MLPilot Configuration"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///mlpilot.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    ARTIFACT_FOLDER = os.environ.get("ARTIFACT_FOLDER", "artifacts")
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50MB

    # Security
    ALLOWED_EXTENSIONS = {"csv", "zip"}
    ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff"}
    MAX_DATASET_ROWS = 500_000
    MAX_DATASET_COLS = 2_000

    # Plans
    PLANS = {
        "free": {
            "name": "Free",
            "max_datasets": 3,
            "max_training_runs": 10,
            "max_optimization_runs": 0,
            "max_downloads": 5,
            "max_dataset_size_mb": 10,
            "allow_cnn": False,
            "allow_advanced_optimization": False,
            "allow_unsupervised": True,
        },
        "trial": {
            "name": "Free Trial",
            "max_datasets": 5,
            "max_training_runs": 20,
            "max_optimization_runs": 5,
            "max_downloads": 10,
            "max_dataset_size_mb": 25,
            "allow_cnn": True,
            "allow_advanced_optimization": True,
            "allow_unsupervised": True,
        },
        "pro_weekly": {
            "name": "Pro Weekly",
            "max_datasets": 50,
            "max_training_runs": 200,
            "max_optimization_runs": 100,
            "max_downloads": 200,
            "max_dataset_size_mb": 100,
            "allow_cnn": True,
            "allow_advanced_optimization": True,
            "allow_unsupervised": True,
        },
        "pro_monthly": {
            "name": "Pro Monthly",
            "max_datasets": 200,
            "max_training_runs": 1_000,
            "max_optimization_runs": 500,
            "max_downloads": 1_000,
            "max_dataset_size_mb": 500,
            "allow_cnn": True,
            "allow_advanced_optimization": True,
            "allow_unsupervised": True,
        }
    }

    DEMO_PAYMENT = os.environ.get("DEMO_PAYMENT", "True").lower() in ("true", "1", "yes")