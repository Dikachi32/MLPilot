"""Image dataset pipeline and CNN training."""
import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

class ImageDatasetLoader:
    """Load and preprocess image datasets from folders or CSV with image paths."""

    def __init__(self, img_size: Tuple[int, int] = (128, 128)):
        self.img_size = img_size
        self.label_encoder = LabelEncoder()

    def load_from_paths(self, paths: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        images = []
        valid_labels = []
        for p, lbl in zip(paths, labels):
            if not os.path.exists(p):
                continue
            try:
                img = Image.open(p).convert("RGB").resize(self.img_size)
                arr = np.array(img) / 255.0
                images.append(arr)
                valid_labels.append(lbl)
            except Exception:
                continue
        if not images:
            raise ValueError("No valid images found.")
        X = np.array(images)
        y = self.label_encoder.fit_transform(valid_labels)
        return X, y

    def save_encoder(self, path: str):
        joblib.dump(self.label_encoder, path)


class CNNTrainer:
    """Lightweight CNN for image classification using TensorFlow/Keras."""

    @staticmethod
    def build_model(input_shape: Tuple[int, ...], num_classes: int) -> Any:
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers

            model = keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv2D(32, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid"),
            ])

            loss = "sparse_categorical_crossentropy" if num_classes > 2 else "binary_crossentropy"
            model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
            return model
        except ImportError:
            raise ImportError("TensorFlow is required for CNN training. Install with: pip install tensorflow")

    @staticmethod
    def train(model: Any, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray, epochs: int = 10, batch_size: int = 32) -> Dict:
        try:
            import tensorflow as tf
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            return {
                "model": model,
                "history": history.history,
                "final_train_acc": round(history.history["accuracy"][-1] * 100, 2),
                "final_val_acc": round(history.history["val_accuracy"][-1] * 100, 2),
            }
        except Exception as e:
            return {"error": str(e)}