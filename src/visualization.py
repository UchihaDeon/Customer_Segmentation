import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


def plot_clusters(df, features):
    if "Cluster" not in df.columns:
        st.warning("⚠️ No 'Cluster' column found in dataset.")
        return

    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x=features[0],
        y=features[1],
        hue="Cluster",
        palette="Set2",
        ax=ax
    )
    ax.set_title("Customer Segmentation Clusters")
    st.pyplot(fig)


def plot_silhouette_scores(df, features, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(df[features])
        score = silhouette_score(df[features], labels)
        scores.append((k, score))

    scores_df = pd.DataFrame(scores, columns=["k", "Silhouette"])
    fig, ax = plt.subplots()
    sns.lineplot(data=scores_df, x="k", y="Silhouette", marker="o", ax=ax)
    ax.set_title("Silhouette Score vs Number of Clusters")
    st.pyplot(fig)


def plot_cluster_profiles(df, features):
    if "Cluster" not in df.columns:
        st.warning("⚠️ No 'Cluster' column found in dataset.")
        return

    profiles = df.groupby("Cluster")[features].mean()
    fig, ax = plt.subplots()
    profiles.plot(kind="bar", ax=ax)
    ax.set_title("Cluster Profiles (Average Feature Values)")
    st.pyplot(fig)