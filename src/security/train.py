import pandas as pd
import numpy as np
import os
import joblib
import time
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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

class SecurityModelTrainer:
    def __init__(self, data_dir='data/processed', model_dir='models/security'):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        print("Loading processed data for security models...")
        self.X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv')).values
        self.y_train = pd.read_csv(os.path.join(data_dir, 'y_sec_train.csv'))['Label'].values
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Compute class weights for imbalanced dataset
        self.classes = np.unique(self.y_train)
        weights = compute_class_weight(class_weight='balanced', classes=self.classes, y=self.y_train)
        self.class_weight_dict = dict(zip(self.classes, weights))
        print("Computed class weights:", self.class_weight_dict)

    def train_logistic_regression(self):
        print("Training Logistic Regression...")
        start = time.time()
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(self.X_train, self.y_train)
        print(f"LogReg trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'logistic.joblib'))

    def train_svm(self):
        print("Training SVM (using RBF kernel)...")
        start = time.time()
        model = SVC(kernel='rbf', class_weight='balanced', probability=True)
        # Note: SVM can be very slow on 80K samples. If this is an issue, we could downsample or use LinearSVC.
        # But we need probability=True for soft voting.
        model.fit(self.X_train, self.y_train)
        print(f"SVM trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'svm.joblib'))

    def train_random_forest(self):
        print("Training Random Forest (via XGBoost)...")
        start = time.time()
        # Calculate sample weights for XGBoost
        sample_weights = np.array([self.class_weight_dict[y] for y in self.y_train])
        model = XGBRFClassifier(n_estimators=100, max_depth=10, tree_method='hist', device=self.device)
        model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print(f"RF trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'rf.joblib'))

    def train_gradient_boosting(self):
        print("Training Gradient Boosting (via XGBoost)...")
        start = time.time()
        sample_weights = np.array([self.class_weight_dict[y] for y in self.y_train])
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, tree_method='hist', device=self.device)
        model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
        print(f"GBM trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'gbm.joblib'))

    def train_extra_trees(self):
        print("Training Extra Trees (via LightGBM)...")
        start = time.time()
        model = LGBMClassifier(n_estimators=100, extra_trees=True, class_weight='balanced', device='cpu')
        model.fit(self.X_train, self.y_train)
        print(f"Extra Trees trained in {time.time()-start:.2f}s")
        joblib.dump(model, os.path.join(self.model_dir, 'extra_trees.joblib'))

    def train_mlp(self):
        print("Training MLP Classifier (via PyTorch)...")
        start = time.time()
        
        input_dim = self.X_train.shape[1]
        output_dim = len(self.classes)
        hidden_dim = 128
        batch_size = 512
        epochs = 20
        
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_train, dtype=torch.long)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device)
        
        # Weighted loss function for class imbalance
        weight_tensor = torch.tensor([self.class_weight_dict[c] for c in sorted(self.class_weight_dict.keys())], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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
        

        wrapped_model = PyTorchClassifierWrapper(model, self.device, output_dim)
        joblib.dump(wrapped_model, os.path.join(self.model_dir, 'mlp_wrapper.joblib'))

    def run_all(self):
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.train_extra_trees()
        self.train_mlp()
        # self.train_svm() # SVM is skipped or deferred if it takes too long. Uncomment to train.
        
        print("Security models trained successfully.")

if __name__ == "__main__":
    trainer = SecurityModelTrainer()
    trainer.run_all()
