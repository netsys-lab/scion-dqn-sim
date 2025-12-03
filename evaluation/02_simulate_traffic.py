#!/usr/bin/env python3
"""
Step 2: Simulate 28 days of traffic between source and destination
Based on evaluation_selective_probing/02_simulate_traffic.py
"""

import os
import pickle
import json
import numpy as np
from datetime import datetime, timedelta


class TrafficSimulator:
    """Simulates realistic traffic patterns"""
    
    def __init__(self, src_as, dst_as, paths):
        self.src_as = src_as
        self.dst_as = dst_as
        self.paths = paths
        self.current_time = 0  # in minutes
        
    def generate_diurnal_rate(self, hour):
        """Generate traffic rate based on hour of day"""
        # Dual peak pattern (morning and evening)
        base_rate = 100  # flows per hour
        
        # Morning peak (8-10 AM)
        morning_factor = np.exp(-((hour - 9) ** 2) / 8)
        
        # Evening peak (6-8 PM)
        evening_factor = np.exp(-((hour - 19) ** 2) / 8)
        
        # Night reduction
        night_factor = 0.3 if (hour < 6 or hour > 22) else 1.0
        
        rate_factor = 1 + 2 * (morning_factor + evening_factor)
        return base_rate * rate_factor * night_factor
    
    def generate_flow_size(self):
        """Generate realistic flow sizes (bytes)"""
        # Mixture of flow types
        flow_type = np.random.choice(['mice', 'medium', 'elephant'], p=[0.7, 0.25, 0.05])
        
        if flow_type == 'mice':
            # Small flows (web requests, control messages)
            size = np.random.exponential(100 * 1024)  # 100 KB average
        elif flow_type == 'medium':
            # Medium flows (file transfers, video chunks)
            size = np.random.lognormal(np.log(10 * 1024 * 1024), 1)  # ~10 MB
        else:
            # Elephant flows (large transfers, backups)
            size = np.random.lognormal(np.log(1024 * 1024 * 1024), 0.5)  # ~1 GB
        
        return max(1024, int(size))  # At least 1 KB
    
    def generate_flow_rate(self, size_bytes):
        """Generate requested rate based on flow size"""
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb < 1:
            # Small flows want low latency, moderate rate
            rate = np.random.uniform(1, 10)
        elif size_mb < 100:
            # Medium flows want good rate
            rate = np.random.uniform(10, 100)
        else:
            # Large flows want maximum rate
            rate = np.random.uniform(50, 500)
        
        return rate
    
    def simulate_day(self, day_number):
        """Simulate one day of traffic"""
        flows = []
        
        for hour in range(24):
            # Get hourly rate
            hourly_rate = self.generate_diurnal_rate(hour)
            
            # Add weekly variation
            weekday = day_number % 7
            if weekday in [5, 6]:  # Weekend
                hourly_rate *= 0.6
            
            # Generate flows for this hour
            n_flows = np.random.poisson(hourly_rate)
            
            for _ in range(n_flows):
                # Random minute within the hour
                minute = np.random.randint(0, 60)
                start_time = day_number * 24 * 60 + hour * 60 + minute
                
                # Generate flow characteristics
                size = self.generate_flow_size()
                rate = self.generate_flow_rate(size)
                
                # Duration based on size and rate
                duration = (size * 8) / (rate * 1e6)  # seconds
                duration_min = duration / 60
                
                flow = {
                    'flow_id': len(flows),
                    'source': self.src_as,
                    'destination': self.dst_as,
                    'start_time': start_time,
                    'end_time': start_time + duration_min,
                    'size_bytes': size,
                    'requested_rate_mbps': rate,
                    'flow_type': self._classify_flow(size)
                }
                
                flows.append(flow)
        
        return flows
    
    def _classify_flow(self, size_bytes):
        """Classify flow by size"""
        size_mb = size_bytes / (1024 * 1024)
        if size_mb < 1:
            return 'mice'
        elif size_mb < 100:
            return 'medium'
        else:
            return 'elephant'
    
    def simulate_period(self, n_days):
        """Simulate traffic for n days"""
        print(f"Simulating {n_days} days of traffic...")
        
        all_flows = []
        
        for day in range(n_days):
            day_flows = self.simulate_day(day)
            all_flows.extend(day_flows)
            
            if (day + 1) % 7 == 0:
                print(f"  - Completed week {(day + 1) // 7}")
        
        return all_flows


def add_ground_truth_performance(flows, paths):
    """Add ground truth performance for flows (for evaluation)"""
    
    # For each flow, determine which path would be optimal
    for flow in flows:
        # Simulate network conditions at flow time
        hour = (flow['start_time'] // 60) % 24
        
        # Network utilization varies by time
        if 8 <= hour <= 20:  # Daytime
            congestion_factor = np.random.uniform(0.7, 0.95)
        else:  # Night
            congestion_factor = np.random.uniform(0.3, 0.6)
        
        # For each path, calculate expected performance
        path_performance = []
        
        for path in paths:
            # Base metrics from static path properties
            base_latency = path['static_metrics']['total_latency']
            base_bandwidth = path['static_metrics']['min_bandwidth']
            
            # Add dynamic effects - make paths VERY diverse to challenge selection
            # Path index affects quality (some paths are consistently better/worse)
            path_idx = int(path['path_id'].split('_')[1])
            path_quality_score = np.sin(path_idx * 0.5) * 0.5 + 0.5  # 0 to 1
            
            # Categorize paths into quality tiers
            if path_quality_score > 0.8:  # Excellent paths (20%)
                quality_multiplier = np.random.uniform(0.9, 1.0)
                latency_multiplier = np.random.uniform(1.0, 1.2)
            elif path_quality_score > 0.6:  # Good paths (20%)
                quality_multiplier = np.random.uniform(0.7, 0.9)
                latency_multiplier = np.random.uniform(1.2, 1.5)
            elif path_quality_score > 0.4:  # Average paths (20%)
                quality_multiplier = np.random.uniform(0.5, 0.7)
                latency_multiplier = np.random.uniform(1.5, 2.0)
            elif path_quality_score > 0.2:  # Poor paths (20%)
                quality_multiplier = np.random.uniform(0.3, 0.5)
                latency_multiplier = np.random.uniform(2.0, 3.0)
            else:  # Very poor paths (20%)
                quality_multiplier = np.random.uniform(0.1, 0.3)
                latency_multiplier = np.random.uniform(3.0, 5.0)
            
            # Apply time-of-day effects differently to different quality paths
            if 8 <= hour <= 20:  # Daytime
                # Good paths degrade less during peak hours
                if path_quality_score > 0.6:
                    congestion_impact = congestion_factor * 0.2
                else:
                    congestion_impact = congestion_factor * 0.8
            else:
                congestion_impact = congestion_factor * 0.3
            
            # Calculate actual metrics
            actual_latency = base_latency * latency_multiplier * (1 + congestion_impact)
            actual_latency += np.random.normal(0, base_latency * 0.05)
            
            actual_bandwidth = base_bandwidth * quality_multiplier * (1 - congestion_impact * 0.5)
            actual_bandwidth = max(10, actual_bandwidth)  # Minimum 10 Mbps
            
            # Success probability heavily influenced by path quality
            base_success = 0.99 if path_quality_score > 0.8 else (0.95 if path_quality_score > 0.4 else 0.85)
            hop_penalty = 0.02 * path['static_metrics']['hop_count']
            success_prob = base_success - hop_penalty - congestion_impact * 0.1
            success_prob = max(0.5, min(0.99, success_prob))
            
            path_performance.append({
                'path_id': path['path_id'],
                'latency': actual_latency,
                'bandwidth': actual_bandwidth,
                'success_prob': success_prob
            })
        
        flow['path_performance'] = path_performance
        
        # Determine optimal paths for different metrics
        flow['optimal_by_latency'] = min(path_performance, key=lambda p: p['latency'])['path_id']
        flow['optimal_by_bandwidth'] = max(path_performance, key=lambda p: p['bandwidth'])['path_id']
        
        # Combined metric (for RL target)
        best_combined = max(
            path_performance,
            key=lambda p: p['success_prob'] * p['bandwidth'] / (p['latency'] + 10)
        )
        flow['optimal_combined'] = best_combined['path_id']


def main():
    """Run traffic simulation"""
    
    if len(sys.argv) != 2:
        print("Usage: python 02_simulate_traffic.py <output_dir>")
        sys.exit(1)
    output_dir = sys.argv[1]
    
    # Load configuration
    with open(os.path.join(output_dir, "dense_config.json"), 'r') as f:
        config = json.load(f)
    
    # Load paths
    with open(os.path.join(output_dir, "dense_paths.pkl"), 'rb') as f:
        paths = pickle.load(f)
    
    src_as = config['source_as']
    dst_as = config['destination_as']
    
    print("=== Traffic Simulation ===")
    print(f"Source AS: {src_as}")
    print(f"Destination AS: {dst_as}")
    print(f"Available paths: {len(paths)}")
    
    # Create simulator
    simulator = TrafficSimulator(src_as, dst_as, paths)
    
    # Simulate 28 days
    flows = simulator.simulate_period(28)
    
    print(f"\nGenerated {len(flows)} flows over 28 days")
    
    # Add ground truth performance
    print("\nCalculating ground truth performance...")
    add_ground_truth_performance(flows, paths)
    
    # Analyze flow statistics
    flow_sizes = [f['size_bytes'] for f in flows]
    flow_rates = [f['requested_rate_mbps'] for f in flows]
    
    mice_flows = sum(1 for f in flows if f['flow_type'] == 'mice')
    medium_flows = sum(1 for f in flows if f['flow_type'] == 'medium')
    elephant_flows = sum(1 for f in flows if f['flow_type'] == 'elephant')
    
    print("\nFlow Statistics:")
    print(f"  - Total flows: {len(flows)}")
    print(f"  - Mice flows: {mice_flows} ({mice_flows/len(flows)*100:.1f}%)")
    print(f"  - Medium flows: {medium_flows} ({medium_flows/len(flows)*100:.1f}%)")
    print(f"  - Elephant flows: {elephant_flows} ({elephant_flows/len(flows)*100:.1f}%)")
    print(f"  - Average flow size: {np.mean(flow_sizes)/1e6:.2f} MB")
    print(f"  - Average requested rate: {np.mean(flow_rates):.1f} Mbps")
    
    # Split into training and evaluation
    total_minutes = 28 * 24 * 60
    train_end = 14 * 24 * 60
    
    train_flows = [f for f in flows if f['start_time'] < train_end]
    eval_flows = [f for f in flows if f['start_time'] >= train_end]
    
    print(f"\nDataset Split:")
    print(f"  - Training flows (day 1-14): {len(train_flows)}")
    print(f"  - Evaluation flows (day 15-28): {len(eval_flows)}")
    
    # Save flows
    print("\nSaving flow data...")
    
    # Save as structured numpy arrays for efficiency
    def flows_to_arrays(flows):
        return {
            'flow_id': np.array([f['flow_id'] for f in flows]),
            'source': np.array([f['source'] for f in flows]),
            'destination': np.array([f['destination'] for f in flows]),
            'start_time': np.array([f['start_time'] for f in flows]),
            'end_time': np.array([f['end_time'] for f in flows]),
            'size_bytes': np.array([f['size_bytes'] for f in flows]),
            'requested_rate_mbps': np.array([f['requested_rate_mbps'] for f in flows]),
            'flow_type': np.array([f['flow_type'] for f in flows]),
            'optimal_by_latency': np.array([f['optimal_by_latency'] for f in flows]),
            'optimal_by_bandwidth': np.array([f['optimal_by_bandwidth'] for f in flows]),
            'optimal_combined': np.array([f['optimal_combined'] for f in flows])
        }
    
    # Save all flows
    np.savez(
        os.path.join(output_dir, "all_flows.npz"),
        **flows_to_arrays(flows)
    )
    
    # Save train/eval splits
    np.savez(
        os.path.join(output_dir, "train_flows.npz"),
        **flows_to_arrays(train_flows)
    )
    
    np.savez(
        os.path.join(output_dir, "eval_flows.npz"),
        **flows_to_arrays(eval_flows)
    )
    
    # Save detailed flow data with performance info
    with open(os.path.join(output_dir, "flows_with_performance.pkl"), 'wb') as f:
        pickle.dump(flows, f)
    
    # Update config
    config['n_flows'] = len(flows)
    config['n_train_flows'] = len(train_flows)
    config['n_eval_flows'] = len(eval_flows)
    config['flow_statistics'] = {
        'mice_percent': mice_flows / len(flows) * 100,
        'medium_percent': medium_flows / len(flows) * 100,
        'elephant_percent': elephant_flows / len(flows) * 100,
        'avg_size_mb': float(np.mean(flow_sizes) / 1e6),
        'avg_rate_mbps': float(np.mean(flow_rates))
    }
    
    with open(os.path.join(output_dir, "dense_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Traffic simulation complete!")
    print(f"\nNext step: Train DQN agent with:")
    print(f"  python 03_train_dqn.py")


if __name__ == "__main__":
    main()