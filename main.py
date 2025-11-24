
from src.preprocessing import load_data, clean_data, scale_features
from src.clustering import run_kmeans
from src.visualization import plot_clusters, plot_silhouette_scores

# Step 1: Preprocessing
df = load_data("data/transactions.csv")
df_clean = clean_data(df)
features = ["PurchaseFrequency", "AmountSpent", "ProductTypes"]
df_scaled = scale_features(df_clean, features)

# Step 2: Clustering
model, labels, score = run_kmeans(df_scaled, features, k=3)
df_scaled["Cluster"] = labels

print("✅ Pipeline complete!")
print(df_scaled.head())
print(f"Silhouette Score: {score:.2f}")

# visualize cluster
plot_clusters(df_scaled, ["PurchaseFrequency", "AmountSpent"])

# Compare silhouette scores for different k values
scores = {}
for k in range(2, 7):
    model, labels, score = run_kmeans(df_scaled, features, k=k)
    # Only compute silhouette if we have at least 2 clusters and less than n_samples
    if len(set(labels)) > 1 and len(set(labels)) < len(df_scaled):
        scores[k] = score
    else:
        print(f"⚠️ Skipping k={k}: invalid number of clusters")
        
plot_silhouette_scores(scores)

