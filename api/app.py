from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import numpy as np
from .routes import router

# Dummy wrappers to allow joblib to unpickle the PyTorch models from __main__
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

app = FastAPI(title="SDN-OSHLD API", version="3.0", description="API for QoS Prediction and Cyberattack Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    # Pre-load the ML models before starting the server
    from .inference import get_inference_service
    print("Pre-loading models...")
    get_inference_service()
    print("Starting Uvicorn server...")
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
