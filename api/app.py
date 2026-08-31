from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .routes import router

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
