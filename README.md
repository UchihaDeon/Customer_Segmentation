
# 🛒 Customer Segmentation Dashboard

A modular data science project for customer segmentation using **K‑Means clustering**.  
This project includes a clean backend pipeline (`src/` modules), an orchestrator (`main.py`), and an interactive **Streamlit dashboard (`app.py`)** for dataset upload, clustering, evaluation, visualization, and download.

---

## 📌 Features
- **Modular pipeline**: Preprocessing, clustering, visualization separated into `src/` modules.
- **Evaluation metrics**: Silhouette score calculation and comparison across multiple `k` values.
- **Visualizations**:
  - Scatter plots of clusters
  - Line plots of silhouette scores
- **Streamlit dashboard**:
  - Upload your own dataset or use the sample one
  - Choose number of clusters (`k`) interactively
  - Tabs for Dataset, Evaluation, Visualization
  - Download clustered dataset as CSV
- **Reproducible workflow**: Clean code, professional documentation, and requirements file.

---

## 📂 Project Structure
```
Customer_Segmentation/
│
├── data/
│   └── transactions.csv        # Sample dataset (50 customers)
│
├── src/
│   ├── preprocessing.py        # Data loading, cleaning, scaling
│   ├── clustering.py           # K-Means clustering + silhouette score
│   └── visualization.py        # Cluster plots + silhouette score plots
│
├── main.py                     # Orchestrates pipeline (backend run)
├── app.py                      # Streamlit dashboard
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

---

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/UchihaDeon/Customer_Segmentation.git
   cd Customer_Segmentation
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Run backend pipeline
```bash
python main.py
```
This runs preprocessing, clustering, evaluation, and visualization in sequence.

### Run Streamlit dashboard
```bash
streamlit run app.py
```
Open the provided local URL in your browser to interact with the dashboard.

---

## 📊 Example Outputs
- **Cluster Scatter Plot**: Customers grouped by purchase frequency and amount spent.
- **Silhouette Score Plot**: Comparison of clustering quality across different `k` values.
- **Downloadable Results**: Export clustered dataset as CSV.

---

## 📑 Requirements
- Python 3.9+
- pandas
- scikit-learn
- matplotlib
- seaborn
- streamlit

Install all with:
```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements
- Add more customer features (Age, Region, Loyalty Score).
- Dynamic feature selection in dashboard.
- Elbow method plot for optimal `k`.
- Cluster profile summaries (average spend, frequency, product types).
- Download evaluation metrics as CSV.

---

## 👨‍💻 Author
**Deon**  
BCA Undergraduate | Full‑Stack Developer | Data Science Intern  
Focused on building modular, user‑centric, and professionally documented projects.
