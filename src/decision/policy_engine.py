class PolicyEngine:
    def __init__(self, attack_threshold=0.85):
        self.attack_threshold = attack_threshold
        
    def evaluate_flow(self, security_pred_class, security_prob, qos_preds):
        """
        Takes predictions from Security and QoS models and outputs a network action.
        
        Args:
            security_pred_class (int): Predicted class label (0 is typically Normal)
            security_prob (float): Confidence/Probability of the prediction
            qos_preds (dict): Dictionary of predicted QoS metrics
            
        Returns:
            dict: Action decision and associated parameters
        """
        # Assuming label '0' or 'Normal' is the benign class. We'll use 0 for Normal.
        # But for robustness, we check if the class represents an attack (anything > 0 usually).
        
        is_attack = security_pred_class != 0
        is_high_confidence = security_prob >= self.attack_threshold
        
        decision = {
            'action': None,
            'reason': '',
            'priority': 'Normal'
        }
        
        if is_attack and is_high_confidence:
            if security_pred_class in [1, 2]: # DoS / DDoS
                decision['action'] = 'BLOCK'
                decision['reason'] = 'High confidence DoS/DDoS detected'
                decision['priority'] = 'High'
            elif security_pred_class == 4: # BruteForce
                decision['action'] = 'RATE LIMIT'
                decision['reason'] = 'BruteForce attack detected'
                decision['priority'] = 'High'
            else:
                decision['action'] = 'MIRROR'
                decision['reason'] = 'Suspicious attack signature (PortScan/Botnet) - Mirror for analysis'
                decision['priority'] = 'Medium'
        else:
            # Benign or low confidence attack (treat as benign but monitor)
            latency_pred = qos_preds.get('latency', 0)
            throughput_pred = qos_preds.get('throughput', 0)
            
            if latency_pred > 100: # High latency prediction
                decision['action'] = 'REROUTE'
                decision['reason'] = 'Predicted QoS violation (High Latency)'
            elif throughput_pred < 10: # Low throughput
                decision['action'] = 'PRIORITIZE'
                decision['reason'] = 'Predicted QoS violation (Low Throughput)'
                decision['priority'] = 'High'
            else:
                decision['action'] = 'ALLOW'
                decision['reason'] = 'Flow is benign and meets QoS SLA'
                
        return decision

if __name__ == "__main__":
    engine = PolicyEngine()
    
    # Test cases
    print("Test 1: DDoS Attack")
    print(engine.evaluate_flow(security_pred_class=2, security_prob=0.95, qos_preds={'latency': 150}))
    
    print("\nTest 2: Normal Flow, High Latency")
    print(engine.evaluate_flow(security_pred_class=0, security_prob=0.99, qos_preds={'latency': 120, 'throughput': 50}))
    
    print("\nTest 3: Normal Flow, Good QoS")
    print(engine.evaluate_flow(security_pred_class=0, security_prob=0.99, qos_preds={'latency': 20, 'throughput': 100}))
