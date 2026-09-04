"""Unsupervised learning model registry and training."""
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, Any, List

class UnsupervisedEngine:
    """Train and evaluate unsupervised models."""

    @staticmethod
    def run_kmeans(X: Any, n_clusters: int = 3) -> Dict:
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        return {
            "model": model,
            "labels": labels,
            "n_clusters": n_clusters,
            "inertia": round(model.inertia_, 2),
            "silhouette": round(score, 4) if score is not None else None,
        }

    @staticmethod
    def run_hierarchical(X: Any, n_clusters: int = 3) -> Dict:
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else None
        return {
            "model": model,
            "labels": labels,
            "n_clusters": n_clusters,
            "silhouette": round(score, 4) if score is not None else None,
        }

    @staticmethod
    def run_pca(X: Any, n_components: float = 0.95) -> Dict:
        model = PCA(n_components=n_components, random_state=42)
        X_reduced = model.fit_transform(X)
        return {
            "model": model,
            "X_reduced": X_reduced,
            "n_components": model.n_components_,
            "explained_variance_ratio": [round(v, 4) for v in model.explained_variance_ratio_],
            "total_variance": round(sum(model.explained_variance_ratio_) * 100, 2),
        }

    @staticmethod
    def auto_cluster(X: Any, max_k: int = 10) -> List[Dict]:
        """Try multiple k values and return results."""
        results = []
        for k in range(2, max_k + 1):
            if len(X) <= k:
                break
            res = UnsupervisedEngine.run_kmeans(X, n_clusters=k)
            results.append({"k": k, **res})
        return results