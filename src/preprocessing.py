# src/preprocessing.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    """Load transactional dataset from CSV."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and drop duplicates."""
    df = df.drop_duplicates()
    df = df.fillna(0)  # simple strategy, can be improved
    return df

def scale_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Normalize selected features using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])
    return df_scaled