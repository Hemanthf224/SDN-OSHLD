import numpy as np

class EnergyOptimizer:
    def __init__(self, base_power_w=4580, min_active_links=2, total_links=10):
        self.base_power_w = base_power_w
        self.current_power_w = base_power_w
        self.min_active_links = min_active_links
        self.total_links = total_links
        self.active_links = total_links
        
    def link_consolidation(self, current_traffic_mbps, max_link_capacity_mbps=10000):
        """
        Determine how many links can be put to sleep based on current traffic volume.
        """
        required_links = int(np.ceil(current_traffic_mbps / max_link_capacity_mbps))
        # Ensure minimum redundancy
        self.active_links = max(self.min_active_links, min(required_links + 1, self.total_links))
        return self.active_links

    def sleep_scheduling(self):
        """
        Calculate power savings by putting unused links into a low-power state.
        Assumes each active link consumes 150W and sleep state consumes 10W.
        """
        sleeping_links = self.total_links - self.active_links
        # Base router chassis power + active link power + sleep link power
        chassis_power = self.base_power_w - (self.total_links * 150)
        self.current_power_w = chassis_power + (self.active_links * 150) + (sleeping_links * 10)
        
        savings_pct = ((self.base_power_w - self.current_power_w) / self.base_power_w) * 100
        return self.current_power_w, savings_pct

    def adaptive_rate_scaling(self, flow_priority, current_rate):
        """
        Scale down rate for low priority background flows during high energy states.
        """
        if flow_priority == 'Low':
            return current_rate * 0.5  # Reduce by 50%
        elif flow_priority == 'Medium':
            return current_rate * 0.8
        return current_rate

    def optimize(self, total_network_traffic_mbps):
        """Run full energy optimization cycle"""
        active_links = self.link_consolidation(total_network_traffic_mbps)
        power, savings = self.sleep_scheduling()
        
        return {
            'active_links': active_links,
            'sleeping_links': self.total_links - active_links,
            'current_power_w': power,
            'power_saved_pct': savings
        }

if __name__ == "__main__":
    optimizer = EnergyOptimizer()
    print("Initial Power:", optimizer.base_power_w, "W")
    
    # Simulate low traffic (e.g. at night)
    print("\nLow Traffic Scenario (5000 Mbps):")
    result = optimizer.optimize(5000)
    print(result)
    
    # Simulate high traffic
    print("\nHigh Traffic Scenario (45000 Mbps):")
    result = optimizer.optimize(45000)
    print(result)
