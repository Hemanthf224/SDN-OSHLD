import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class SyntheticDataGenerator:
    def __init__(self, num_samples=100000, output_dir='data/synthetic'):
        self.num_samples = num_samples
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Distribution of classes (Botnet should be very small ~ 0.07% as per report)
        self.class_distribution = {
            'Normal': 0.60,
            'DoS': 0.15,
            'DDoS': 0.10,
            'PortScan': 0.10,
            'BruteForce': 0.0493,
            'Botnet': 0.0007
        }
        
    def generate_flow_features(self, class_name, count):
        """Generate network flow features based on the class behavior."""
        if class_name == 'Normal':
            packet_count = np.random.normal(50, 20, count)
            bytes_count = packet_count * np.random.normal(500, 100, count)
            duration = np.random.exponential(1.5, count)
        elif class_name in ['DoS', 'DDoS']:
            packet_count = np.random.normal(5000, 1000, count)
            bytes_count = packet_count * np.random.normal(64, 10, count)
            duration = np.random.uniform(0.1, 0.5, count)
        elif class_name == 'PortScan':
            packet_count = np.random.normal(2, 0.5, count)
            bytes_count = packet_count * np.random.normal(40, 5, count)
            duration = np.random.uniform(0.01, 0.05, count)
        elif class_name == 'BruteForce':
            packet_count = np.random.normal(20, 5, count)
            bytes_count = packet_count * np.random.normal(200, 50, count)
            duration = np.random.uniform(2.0, 5.0, count)
        elif class_name == 'Botnet':
            # Low and slow / heartbeat traffic
            packet_count = np.random.normal(5, 1, count)
            bytes_count = packet_count * np.random.normal(100, 20, count)
            duration = np.random.exponential(10.0, count)
            
        # Add noise
        packet_count = np.clip(packet_count, 1, None)
        bytes_count = np.clip(bytes_count, 40, None)
        duration = np.clip(duration, 0.001, None)
        
        # QoS Metrics generation based on traffic profile
        bandwidth = (bytes_count * 8) / duration / 1e6 # Mbps
        
        if class_name == 'Normal':
            latency = np.random.normal(20, 5, count)
            jitter = np.random.normal(2, 1, count)
            packet_loss = np.random.uniform(0, 0.01, count)
        else:
            # Attacks cause network degradation
            latency = np.random.normal(150, 50, count)
            jitter = np.random.normal(20, 10, count)
            packet_loss = np.random.uniform(0.05, 0.2, count)
            
        throughput = bandwidth * (1 - packet_loss)
        energy_efficiency = throughput / (latency + 1)
        qos_score = (throughput * 0.5) - (latency * 0.3) - (jitter * 0.2)
        flow_completion = duration + (packet_loss * 2.0)
        p95_latency = latency * 1.5
        p99_latency = latency * 2.0
        total_energy = (packet_count * 0.001) + (duration * 0.5)

        return pd.DataFrame({
            'packet_count': packet_count,
            'bytes': bytes_count,
            'duration': duration,
            'bandwidth': bandwidth,
            
            # QoS targets
            'latency': latency,
            'jitter': jitter,
            'packet_loss': packet_loss,
            'throughput': throughput,
            'energy_efficiency': energy_efficiency,
            'qos_score': qos_score,
            'flow_completion': flow_completion,
            'p95_latency': p95_latency,
            'p99_latency': p99_latency,
            'total_energy': total_energy,
            
            # Labels
            'Label': class_name
        })

    def generate(self):
        print(f"Generating {self.num_samples} synthetic network flow samples...")
        dfs = []
        for class_name, proportion in self.class_distribution.items():
            count = int(self.num_samples * proportion)
            print(f" - Generating {count} {class_name} flows")
            df_class = self.generate_flow_features(class_name, count)
            dfs.append(df_class)
            
        full_df = pd.concat(dfs, ignore_index=True)
        # Shuffle
        full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_path = os.path.join(self.output_dir, 'synthetic_network_data.csv')
        full_df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        return full_df

if __name__ == "__main__":
    generator = SyntheticDataGenerator(num_samples=100000)
    generator.generate()
