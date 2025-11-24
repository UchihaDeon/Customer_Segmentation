# src/clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans(df: pd.DataFrame, features: list, k: int = 3):
    """Run K-Means clustering and return model, labels, and silhouette score."""
    X = df[features]
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    return model, labels, score