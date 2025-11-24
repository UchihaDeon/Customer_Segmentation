
import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(df, features, label_col="Cluster"):
    """Scatter plot of clusters using two main features."""
    plt.figure(figsize=(8,6))
    sns.scatterplot(
        x=df[features[0]],
        y=df[features[1]],
        hue=df[label_col],
        palette="Set2",
        s=100
    )
    plt.title("Customer Segmentation Clusters")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend(title="Cluster")
    plt.show()

def plot_silhouette_scores(scores):
    """Line plot of silhouette scores for different k values."""
    plt.figure(figsize=(8,6))
    sns.lineplot(x=list(scores.keys()), y=list(scores.values()), marker="o")
    plt.title("Silhouette Scores by Number of Clusters (k)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()