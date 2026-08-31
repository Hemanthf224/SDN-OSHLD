import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    def __init__(self, target_col='Label'):
        self.target_col = target_col
        self.scaler = RobustScaler()
        self.label_encoder = LabelEncoder()
        
    def fit_transform(self, df):
        """
        Full pipeline:
        1. Handle missing values
        2. Encode categorical data
        3. Handle outliers using RobustScaler
        """
        df = df.copy()
        
        # 1. Handle missing values (fill with median for numerical, mode for categorical)
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        # 2. Separate features and target
        if self.target_col in df.columns:
            y = df[self.target_col]
            X = df.drop(columns=[self.target_col])
        else:
            X = df
            y = None
            
        # 3. Categorical Encoding (if any categorical features exist)
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        # In SDN-OSHLD, protocols might be categorical
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            
        # 4. Handle Outliers and Scale Features using RobustScaler
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns)
        
        # 5. Encode Target
        if y is not None:
            y_encoded = self.label_encoder.fit_transform(y)
            return X_scaled, y_encoded, self.label_encoder.classes_
        
        return X_scaled, None, None
        
    def transform(self, df):
        """Transform new data based on fitted parameters"""
        df = df.copy()
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
                
        if self.target_col in df.columns:
            X = df.drop(columns=[self.target_col])
        else:
            X = df
            
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        # Note: in a production setting we should save the label encoders per column 
        # to ensure consistent transformation. For simplicity, we assume numeric features 
        # from our synthetic generator mostly.
        
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        return X_scaled

    def save_preprocessor(self, path):
        """Save the fitted scaler and label encoder"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.save({'scaler': self.scaler, 'label_encoder': self.label_encoder}, path)
        
    def load_preprocessor(self, path):
        """Load fitted parameters"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']

def get_train_test_split(X, y, test_size=0.2):
    """Returns stratified train and test sets"""
    # 80/20 stratified split as per SDN-OSHLD paper
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

if __name__ == "__main__":
    # Test on synthetic data
    data_path = 'data/synthetic/synthetic_network_data.csv'
    if os.path.exists(data_path):
        print(f"Loading synthetic data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # For security ML, target is Label.
        # But we also have QoS metrics. We should separate QoS metrics and Attack labels.
        qos_columns = [
            'latency', 'jitter', 'packet_loss', 'throughput', 
            'energy_efficiency', 'qos_score', 'flow_completion', 
            'p95_latency', 'p99_latency', 'total_energy'
        ]
        
        # Security Labels
        security_target = 'Label'
        
        # Features
        feature_cols = [c for c in df.columns if c not in qos_columns and c != security_target]
        print(f"Features: {feature_cols}")
        print(f"QoS Targets: {qos_columns}")
        
        X = df[feature_cols]
        y_sec = df[security_target]
        Y_qos = df[qos_columns]
        
        # Preprocess features
        preprocessor = DataPreprocessor(target_col=None)
        
        # We handle X independently
        X_scaled = preprocessor.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Label Encode Security Target
        le = LabelEncoder()
        y_sec_encoded = le.fit_transform(y_sec)
        
        # Split Data (Stratified by attack type)
        X_train, X_test, y_sec_train, y_sec_test, y_qos_train, y_qos_test = train_test_split(
            X_scaled, y_sec_encoded, Y_qos, test_size=0.2, stratify=y_sec_encoded, random_state=42
        )
        
        print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        
        # Save datasets
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        
        pd.DataFrame(y_sec_train, columns=['Label']).to_csv('data/processed/y_sec_train.csv', index=False)
        pd.DataFrame(y_sec_test, columns=['Label']).to_csv('data/processed/y_sec_test.csv', index=False)
        
        y_qos_train.to_csv('data/processed/y_qos_train.csv', index=False)
        y_qos_test.to_csv('data/processed/y_qos_test.csv', index=False)
        
        # Save preprocessor
        preprocessor.save_preprocessor('models/preprocessor.joblib')
        
        print("Preprocessing complete. Data saved to data/processed/")
    else:
        print("Synthetic dataset not found. Please run synthetic_generator.py first.")
