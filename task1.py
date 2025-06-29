import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Output file path
OUTPUT_FILE = Path("output/processed_data.csv")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

def load_data():
    import logging
    import pandas as pd

    logging.info("📥 Loading real dataset from CSV...")
    df = pd.read_csv("input/real_dataset.csv")
    logging.info(f"✅ Raw Data:\n{df}")
    return df


def preprocess_data(df):
    logging.info("🧹 Preprocessing: Handling missing values...")
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    return df

def transform_data(df):
    logging.info("🔁 Transforming: Encoding and Scaling...")
    
    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])  # F = 0, M = 1

    # Scale Age and Income
    scaler = StandardScaler()
    df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
    
    return df

def save_data(df):
    logging.info(f"💾 Saving processed data to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info("✅ File saved successfully!")

def main():
    df = load_data()
    df = preprocess_data(df)
    df = transform_data(df)
    logging.info(f"✅ Final Processed Data:\n{df}")
    save_data(df)

if __name__ == "__main__":
    main()
