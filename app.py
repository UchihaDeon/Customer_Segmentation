import streamlit as st
import pandas as pd
from src.preprocessing import load_data, clean_data, scale_features
from src.clustering import run_kmeans
from src.visualization import plot_clusters, plot_silhouette_scores, plot_cluster_profiles

st.title("🛒 Customer Segmentation Dashboard")

# Upload or use default dataset
uploaded_file = st.file_uploader("Upload your transactions CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data("data/transactions.csv")

# Preprocess
df_clean = clean_data(df)
features = ["PurchaseFrequency", "AmountSpent", "ProductTypes"]
df_scaled = scale_features(df_clean, features)

# Select number of clusters
k = st.slider("Select number of clusters (k)", 2, 10, 3)

# Run clustering
model, labels, score = run_kmeans(df_scaled, features, k)
df_scaled["Cluster"] = labels

# Tabs for cleaner interface
tab1, tab2, tab3 = st.tabs(["📂 Dataset", "📊 Evaluation", "🎨 Visualization"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_scaled.head())

    # Download clustered dataset
    csv = df_scaled.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Clustered Dataset",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )

with tab2:
    st.subheader("Evaluation Metrics")
    st.write(f"Silhouette Score: **{score:.2f}**")
    st.write("Cluster Counts:")
    st.write(df_scaled["Cluster"].value_counts())

with tab3:
    st.subheader("Cluster Visualization")
    plot_clusters(df_scaled, ["PurchaseFrequency", "AmountSpent"])

    st.subheader("Silhouette Score Comparison")
    plot_silhouette_scores(df_scaled, ["PurchaseFrequency", "AmountSpent"], max_k=10)

    st.subheader("Cluster Profiles")
    plot_cluster_profiles(df_scaled, ["PurchaseFrequency", "AmountSpent", "ProductTypes"])