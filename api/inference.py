import os
import joblib
import pandas as pd
import numpy as np

# We need the PyTorch wrappers in scope
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.qos.ensemble import QoSEnsemble
from src.security.ensemble import SecurityEnsemble
from src.decision.policy_engine import PolicyEngine
from src.routing.energy_model import EnergyOptimizer

class InferenceService:
    def __init__(self):
        print("Initializing Inference Service...")
        
        # Load Preprocessor
        preprocessor_data = joblib.load('models/preprocessor.joblib')
        self.scaler = preprocessor_data['scaler']
        self.label_encoder = preprocessor_data['label_encoder']
        
        # Load Ensembles
        self.qos_ensemble = QoSEnsemble()
        self.qos_ensemble.calculate_weights(self.qos_ensemble.evaluate_models()) # Recalculates or could just load the json
        
        self.security_ensemble = SecurityEnsemble()
        self.security_ensemble.calculate_weights(self.security_ensemble.evaluate_models())
        
        # Load Policy Engine & Energy Optimizer
        self.policy_engine = PolicyEngine()
        self.energy_optimizer = EnergyOptimizer()

    def process_flow(self, flow_data):
        """
        End-to-end inference for a single network flow.
        """
        # 1. Preprocess
        df = pd.DataFrame([flow_data])
        X_scaled = self.scaler.transform(df)
        
        # 2. QoS Prediction
        qos_preds_raw = self.qos_ensemble.predict_ensemble(X_scaled)[0]
        qos_cols = ['latency', 'jitter', 'packet_loss', 'throughput', 
                    'energy_efficiency', 'qos_score', 'flow_completion', 
                    'p95_latency', 'p99_latency', 'total_energy']
        qos_dict = dict(zip(qos_cols, qos_preds_raw))
        
        # 3. Security Prediction
        sec_pred_class, sec_probs = self.security_ensemble.predict_ensemble(X_scaled)
        sec_class_idx = sec_pred_class[0]
        sec_prob = np.max(sec_probs[0])
        class_names = ['Botnet', 'BruteForce', 'DDoS', 'DoS', 'Normal', 'PortScan']
        class_name = class_names[int(sec_class_idx)]
        
        # 4. Decision Engine
        decision = self.policy_engine.evaluate_flow(
            security_pred_class=sec_class_idx,
            security_prob=sec_prob,
            qos_preds=qos_dict
        )
        
        # 5. Energy Optimization (Simulated for one flow by treating it as added traffic)
        energy_result = self.energy_optimizer.optimize(total_network_traffic_mbps=5000 + flow_data['bandwidth'])
        
        return {
            'pred_class': int(sec_class_idx),
            'pred_class_name': class_name,
            'pred_conf': float(sec_prob),
            'qos_predictions': {k: float(v) for k, v in qos_dict.items()},
            'action': decision['action'],
            'reason': decision['reason'],
            'priority': decision['priority'],
            'energy_power_w': float(energy_result['current_power_w']),
            'energy_saved_pct': float(energy_result['power_saved_pct'])
        }

# Global singleton
inference_service = None

def get_inference_service():
    global inference_service
    if inference_service is None:
        inference_service = InferenceService()
    return inference_service
