import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import json
import torch

class PyTorchClassifierWrapper:
    def __init__(self, model, device, num_classes):
        self.model = model
        self.device = device
        self.classes_ = np.arange(num_classes)
        
    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

class SecurityEnsemble:
    def __init__(self, data_dir='data/processed', model_dir='models/security'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        print("Loading test data for security evaluation...")
        self.X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv')).values
        self.y_test = pd.read_csv(os.path.join(data_dir, 'y_sec_test.csv'))['Label'].values
        
        print("Loading trained models...")
        self.models = {
            'LogisticRegression': joblib.load(os.path.join(model_dir, 'logistic.joblib')),
            'RandomForest': joblib.load(os.path.join(model_dir, 'rf.joblib')),
            'GradientBoosting': joblib.load(os.path.join(model_dir, 'gbm.joblib')),
            'ExtraTrees': joblib.load(os.path.join(model_dir, 'extra_trees.joblib')),
            'MLP': joblib.load(os.path.join(model_dir, 'mlp_wrapper.joblib'))
        }
        
        self.weights = {}

    def evaluate_models(self):
        """Evaluate each individual model and calculate Accuracy and F1 Score."""
        results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            preds = model.predict(self.X_test)
            
            acc = accuracy_score(self.y_test, preds)
            f1 = f1_score(self.y_test, preds, average='weighted')
            
            # Predict Proba for AUC
            try:
                probs = model.predict_proba(self.X_test)
                auc = roc_auc_score(self.y_test, probs, multi_class='ovr', average='weighted')
            except Exception as e:
                print(f"Warning: Could not calculate AUC for {name}: {e}")
                auc = 0.5
                
            results[name] = {'Accuracy': acc, 'F1': f1, 'AUC': auc}
            print(f"  -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
        return results

    def calculate_weights(self, results):
        """Calculate performance-based weights (Accuracy * F1)"""
        print("Calculating ensemble weights...")
        
        raw_weights = {name: results[name]['Accuracy'] * results[name]['F1'] for name in self.models.keys()}
        total_weight = sum(raw_weights.values())
        
        self.weights = {name: w / total_weight for name, w in raw_weights.items()}
        
        # Save weights
        with open(os.path.join(self.model_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f, indent=4)
        print("Security ensemble weights saved.")

    def predict_ensemble(self, X):
        """Make predictions using soft-voting ensemble"""
        ensemble_probs = None
        
        for name, model in self.models.items():
            probs = model.predict_proba(X)
            weight = self.weights[name]
            
            if ensemble_probs is None:
                ensemble_probs = probs * weight
            else:
                ensemble_probs += probs * weight
                
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        return ensemble_preds, ensemble_probs

    def evaluate_ensemble(self):
        print("Evaluating Soft-Voting Security Ensemble...")
        preds, probs = self.predict_ensemble(self.X_test)
        
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average='weighted')
        auc = roc_auc_score(self.y_test, probs, multi_class='ovr', average='weighted')
        
        print(f"Ensemble -> Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, preds))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, preds))

if __name__ == "__main__":
    ensemble = SecurityEnsemble()
    results = ensemble.evaluate_models()
    ensemble.calculate_weights(results)
    ensemble.evaluate_ensemble()
