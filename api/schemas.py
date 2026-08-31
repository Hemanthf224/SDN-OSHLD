from pydantic import BaseModel
from typing import Dict, Any

class NetworkFlow(BaseModel):
    packet_count: float
    bytes: float
    duration: float
    bandwidth: float

class InferenceRequest(BaseModel):
    features: NetworkFlow

class InferenceResponse(BaseModel):
    pred_class: int
    pred_class_name: str
    pred_conf: float
    qos_predictions: Dict[str, float]
    action: str
    reason: str
    priority: str
    energy_power_w: float
    energy_saved_pct: float
