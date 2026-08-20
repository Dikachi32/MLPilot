"""Model artifact management and packaging."""
import os
import json
import joblib
import zipfile
from typing import Dict, Any
from datetime import datetime

class ArtifactManager:
    def __init__(self, base_folder: str):
        self.base_folder = base_folder
        os.makedirs(base_folder, exist_ok=True)

    def save_model(self, model: Any, name: str, experiment_id: str) -> str:
        folder = os.path.join(self.base_folder, experiment_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{name.replace(' ', '_')}.pkl")
        joblib.dump(model, path)
        return path

    def save_metadata(self, metadata: Dict, experiment_id: str) -> str:
        folder = os.path.join(self.base_folder, experiment_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "metadata.json")
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return path

    def package_all(self, experiment_id: str, model_paths: Dict[str, str],
                    preproc_paths: Dict[str, str], metadata: Dict) -> str:
        """Create a zip package with model + preprocessing + metadata."""
        folder = os.path.join(self.base_folder, experiment_id)
        os.makedirs(folder, exist_ok=True)
        zip_path = os.path.join(folder, f"mlpilot_package_{experiment_id}.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, path in model_paths.items():
                if os.path.exists(path):
                    zf.write(path, os.path.join("models", os.path.basename(path)))
            for name, path in preproc_paths.items():
                if os.path.exists(path):
                    zf.write(path, os.path.join("preprocessing", os.path.basename(path)))
            meta_path = os.path.join(folder, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            zf.write(meta_path, "metadata.json")

        return zip_path