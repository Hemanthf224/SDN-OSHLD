import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import json
import torch

class PyTorchWrapper:
    def __init__(self, model, device, input_dim, output_dim):
        self.model = model
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            preds = self.model(X_t).cpu().numpy()
        return preds

class QoSEnsemble:
    def __init__(self, data_dir='data/processed', model_dir='models/qos'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        
        print("Loading test data for evaluation & ensemble weighting...")
        self.X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv')).values
        self.y_test = pd.read_csv(os.path.join(data_dir, 'y_qos_test.csv'))
        self.target_cols = self.y_test.columns
        self.y_test = self.y_test.values
        
        # Load models
        print("Loading trained models...")
        self.models = {
            'Ridge': joblib.load(os.path.join(model_dir, 'ridge.joblib')),
            'RandomForest': joblib.load(os.path.join(model_dir, 'rf.joblib')),
            'GradientBoosting': joblib.load(os.path.join(model_dir, 'gbm.joblib')),
            'ExtraTrees': joblib.load(os.path.join(model_dir, 'extra_trees.joblib')),
            'MLP': joblib.load(os.path.join(model_dir, 'mlp_wrapper.joblib'))
        }
        
        self.weights = {} # Will hold per-metric weights

    def evaluate_models(self):
        """Evaluate each individual model and calculate R^2, MAE, RMSE"""
        results = {}
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            preds = model.predict(self.X_test)
            
            model_results = {}
            for i, col in enumerate(self.target_cols):
                r2 = r2_score(self.y_test[:, i], preds[:, i])
                mae = mean_absolute_error(self.y_test[:, i], preds[:, i])
                rmse = np.sqrt(mean_squared_error(self.y_test[:, i], preds[:, i]))
                model_results[col] = {'R2': r2, 'MAE': mae, 'RMSE': rmse}
                
            results[name] = model_results
            
            # Print average R2 for quick summary
            avg_r2 = np.mean([res['R2'] for res in model_results.values()])
            print(f"  -> Average R2: {avg_r2:.4f}")
            
        return results

    def calculate_weights(self, results):
        """Calculate performance-based weights (using squared R^2 as per SDN-OSHLD paper)"""
        print("Calculating per-metric ensemble weights...")
        for col in self.target_cols:
            self.weights[col] = {}
            
            # Get R2 scores for this metric across all models
            r2_scores = {name: max(0, results[name][col]['R2']) for name in self.models.keys()}
            
            # Calculate weights based on squared R2 to heavily favor better models
            squared_r2 = {name: score**2 for name, score in r2_scores.items()}
            total_squared_r2 = sum(squared_r2.values())
            
            if total_squared_r2 == 0:
                # Fallback to equal weighting if all models fail
                weight_val = 1.0 / len(self.models)
                self.weights[col] = {name: weight_val for name in self.models.keys()}
            else:
                self.weights[col] = {name: score/total_squared_r2 for name, score in squared_r2.items()}
                
        # Save weights
        with open(os.path.join(self.model_dir, 'ensemble_weights.json'), 'w') as f:
            json.dump(self.weights, f, indent=4)
        print("Ensemble weights saved.")

    def predict_ensemble(self, X):
        """Make predictions using the weighted ensemble"""
        all_preds = {name: model.predict(X) for name, model in self.models.items()}
        ensemble_preds = np.zeros_like(all_preds['Ridge'])
        
        for i, col in enumerate(self.target_cols):
            for name in self.models.keys():
                weight = self.weights[col][name]
                ensemble_preds[:, i] += all_preds[name][:, i] * weight
                
        return ensemble_preds

    def evaluate_ensemble(self):
        print("Evaluating Adaptive Ensemble...")
        preds = self.predict_ensemble(self.X_test)
        
        total_r2 = 0
        for i, col in enumerate(self.target_cols):
            r2 = r2_score(self.y_test[:, i], preds[:, i])
            total_r2 += r2
            print(f"  {col} -> Ensemble R2: {r2:.4f}")
            
        print(f"Overall Average Ensemble R2: {total_r2/len(self.target_cols):.4f}")
        return preds

if __name__ == "__main__":
    ensemble = QoSEnsemble()
    results = ensemble.evaluate_models()
    ensemble.calculate_weights(results)
    ensemble.evaluate_ensemble()
