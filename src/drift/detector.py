import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class DriftDetector:
    def __init__(self, reference_window_size=10000, threshold=0.05, model_dir='models/drift'):
        self.reference_window_size = reference_window_size
        self.threshold = threshold
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Isolation Forest for multivariate drift detection
        self.detector = IsolationForest(n_estimators=100, contamination=self.threshold, random_state=42)
        self.is_fitted = False
        
    def fit_reference(self, X_reference):
        """Train the detector on a stable reference window of network traffic."""
        print(f"Fitting Drift Detector on {len(X_reference)} reference samples...")
        # If X_reference is larger than the window, take the most recent samples
        if len(X_reference) > self.reference_window_size:
            X_reference = X_reference[-self.reference_window_size:]
            
        self.detector.fit(X_reference)
        self.is_fitted = True
        
        joblib.dump(self.detector, os.path.join(self.model_dir, 'isolation_forest.joblib'))
        print("Drift detector fitted and saved.")
        
    def detect_drift(self, X_batch):
        """
        Evaluate new batch of data. 
        Returns True if the proportion of anomalies exceeds the threshold.
        """
        if not self.is_fitted:
            model_path = os.path.join(self.model_dir, 'isolation_forest.joblib')
            if os.path.exists(model_path):
                self.detector = joblib.load(model_path)
                self.is_fitted = True
            else:
                raise ValueError("Drift detector is not fitted and no saved model found.")
                
        # -1 for anomalies, 1 for normal
        preds = self.detector.predict(X_batch)
        
        # Calculate anomaly rate
        anomaly_rate = np.sum(preds == -1) / len(preds)
        
        is_drifting = anomaly_rate > self.threshold
        
        return {
            'is_drifting': is_drifting,
            'anomaly_rate': float(anomaly_rate),
            'threshold': self.threshold
        }

if __name__ == "__main__":
    # Test Drift Detector
    detector = DriftDetector()
    
    # Mock reference data (normal distribution)
    np.random.seed(42)
    X_ref = np.random.normal(0, 1, (10000, 10))
    detector.fit_reference(X_ref)
    
    # Mock normal batch
    X_batch_normal = np.random.normal(0, 1, (1000, 10))
    print("Normal batch test:", detector.detect_drift(X_batch_normal))
    
    # Mock drifted batch (mean shifted)
    X_batch_drift = np.random.normal(2, 1, (1000, 10))
    print("Drifted batch test:", detector.detect_drift(X_batch_drift))
