import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class QoSModelTrainer:
    def __init__(self, data_dir='data/processed', model_dir='models/qos'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("Loading processed data...")
        self.X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).values
        self.y_train = pd.read_csv(os.path.join(data_dir, 'y_qos_train.csv')).values
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def train_ridge(self):
        print("Training Ridge Regression...")
        start = time.time()
        # Ridge inherently supports multi-output in scikit-learn
        model = Ridge(alpha=1.0)
        model.fit(self.X_train, self.y_train)
        print(f"Ridge trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'ridge.joblib'))

    def train_random_forest(self):
        print("Training Random Forest (via XGBoost for GPU)...")
        start = time.time()
        # Using XGBRFRegressor as a proxy for GPU Random Forest
        base_model = XGBRFRegressor(n_estimators=100, max_depth=10, tree_method='hist', device=self.device)
        model = MultiOutputRegressor(base_model)
        model.fit(self.X_train, self.y_train)
        print(f"RF trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'rf.joblib'))

    def train_gradient_boosting(self):
        print("Training Gradient Boosting (via XGBoost)...")
        start = time.time()
        base_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', device=self.device)
        model = MultiOutputRegressor(base_model)
        model.fit(self.X_train, self.y_train)
        print(f"GBM trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'gbm.joblib'))

    def train_extra_trees(self):
        print("Training Extra Trees (via LightGBM)...")
        start = time.time()
        # LightGBM Extra Trees
        base_model = LGBMRegressor(n_estimators=100, extra_trees=True, device='cpu') # lgbm gpu setup on windows is tricky, fallback to cpu
        model = MultiOutputRegressor(base_model)
        model.fit(self.X_train, self.y_train)
        print(f"Extra Trees trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'extra_trees.joblib'))

    def train_mlp(self):
        print("Training MLP Regressor (via PyTorch)...")
        start = time.time()
        
        # Hyperparams
        input_dim = self.X_train.shape[1]
        output_dim = self.y_train.shape[1]
        hidden_dim = 128
        batch_size = 512
        epochs = 20
        
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.float32)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
                
        print(f"MLP trained in {time.time()-start:.2f}s")
        torch.save(model.state_dict(), os.path.join(self.model_dir, 'mlp.pt'))
        

        wrapped_model = PyTorchWrapper(model, self.device, input_dim, output_dim)
        joblib.dump(wrapped_model, os.path.join(self.model_dir, 'mlp_wrapper.joblib'))

    def run_all(self):
        self.train_ridge()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_extra_trees()
        self.train_mlp()
        print("All QoS models trained successfully.")

if __name__ == "__main__":
    trainer = QoSModelTrainer()
    trainer.run_all()
