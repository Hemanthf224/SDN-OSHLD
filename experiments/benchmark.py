import time
import requests
import json
import numpy as np

# Same distribution generator as the frontend
def generate_flow():
    is_normal = np.random.rand() < 0.8
    if is_normal:
        packet_count = np.random.normal(50, 20)
        bytes_count = packet_count * np.random.normal(500, 100)
        duration = np.random.exponential(1.5)
    else:
        packet_count = np.random.normal(5000, 1000)
        bytes_count = packet_count * np.random.normal(64, 10)
        duration = np.random.uniform(0.1, 0.5)
        
    packet_count = max(1, packet_count)
    bytes_count = max(40, bytes_count)
    duration = max(0.001, duration)
    bandwidth = (bytes_count * 8) / duration / 1e6
    
    return {
        "features": {
            "packet_count": packet_count,
            "bytes": bytes_count,
            "duration": duration,
            "bandwidth": bandwidth
        }
    }

def benchmark(url="http://127.0.0.1:8000/api/v1/infer", num_requests=1000):
    print(f"Starting benchmark of {num_requests} requests against {url}...")
    latencies = []
    errors = 0
    
    start_total = time.time()
    
    for i in range(num_requests):
        payload = generate_flow()
        try:
            req_start = time.time()
            response = requests.post(url, json=payload, timeout=2)
            req_end = time.time()
            
            if response.status_code == 200:
                latencies.append((req_end - req_start) * 1000) # convert to ms
            else:
                if errors == 0:
                    print("Error response:", response.text)
                errors += 1
                
        except Exception as e:
            if errors == 0:
                print("Exception:", str(e))
            errors += 1
            
        if (i+1) % 200 == 0:
            print(f"Processed {i+1}/{num_requests} flows...")

    end_total = time.time()
    
    total_time = end_total - start_total
    throughput = num_requests / total_time
    
    if len(latencies) > 0:
        avg_latency = np.mean(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
    else:
        avg_latency = p50 = p95 = p99 = 0
        
    print("\n--- Benchmark Results ---")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Errors: {errors}")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"p50 Latency: {p50:.2f} ms")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")
    print("-------------------------\n")

if __name__ == "__main__":
    # Wait a few seconds to ensure the server is up
    time.sleep(2)
    benchmark()
