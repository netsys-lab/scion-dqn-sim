#!/usr/bin/env python3
"""
Run SCION beaconing simulation and analyze path diversity
"""

import os
import sys
import json
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict, Counter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.scion_simulator import SCIONSimulator
from src.simulation.path_store import InMemoryPathStore

# Get run directory from command line or use latest
if len(sys.argv) > 1:
    run_dir = sys.argv[1]
else:
    # Find latest run directory
    dirs = [d for d in os.listdir('.') if d.startswith('run_')]
    run_dir = sorted(dirs)[-1]

print(f"Using run directory: {run_dir}")

# Load topology
topology_file = os.path.join(run_dir, "scion_topology.json")
with open(topology_file, 'r') as f:
    topology_data = json.load(f)

# Convert back to NetworkX graph
G = nx.node_link_graph(topology_data['graph'])
topology = {
    'graph': G,
    'isds': topology_data['isds'],
    'core_ases': topology_data['core_ases']
}

print(f"\nLoaded topology with {G.number_of_nodes()} ASes")

# Run beaconing simulation
print("\nRunning SCION beaconing simulation...")
print("This will take a few minutes for 50 ASes...")

simulator = SCIONSimulator(topology)
path_store = InMemoryPathStore()

# Run for sufficient time to propagate all beacons
simulation_time = 600  # 10 minutes
print(f"Simulating for {simulation_time} seconds...")
simulator.run(until=simulation_time, path_store=path_store)

# Save path store
path_store_file = os.path.join(run_dir, "path_store.pkl")
with open(path_store_file, 'wb') as f:
    pickle.dump(path_store, f)
print(f"Path store saved to: {path_store_file}")

# Analyze path diversity
print("\nAnalyzing path diversity...")

# Count paths between AS pairs
path_counts = defaultdict(int)
path_details = defaultdict(list)

for src_as in G.nodes():
    for dst_as in G.nodes():
        if src_as == dst_as:
            continue
        
        paths = path_store.find_paths(src_as, dst_as)
        if paths:
            path_counts[(src_as, dst_as)] = len(paths)
            path_details[(src_as, dst_as)] = paths

# Find pairs with high path diversity
diverse_pairs = [(pair, count) for pair, count in path_counts.items() if count >= 5]
diverse_pairs.sort(key=lambda x: x[1], reverse=True)

print(f"\nTotal AS pairs with paths: {len(path_counts)}")
print(f"AS pairs with 5+ paths: {len(diverse_pairs)}")

# Select best source-destination pair
if diverse_pairs:
    best_pair, best_count = diverse_pairs[0]
    src_as, dst_as = best_pair
    
    print(f"\nSelected source-destination pair:")
    print(f"  Source AS: {src_as}")
    print(f"  Destination AS: {dst_as}")
    print(f"  Number of paths: {best_count}")
    
    # Analyze path characteristics
    paths = path_details[best_pair]
    hop_counts = []
    latencies = []
    bandwidths = []
    
    for path in paths:
        # Calculate path metrics
        hops = len(path['hops'])
        latency = sum(hop.get('latency', 10) for hop in path['hops'])
        bandwidth = min(hop.get('bandwidth', 1000) for hop in path['hops'])
        
        hop_counts.append(hops)
        latencies.append(latency)
        bandwidths.append(bandwidth)
    
    print(f"\n  Path characteristics:")
    print(f"    Hop counts: min={min(hop_counts)}, max={max(hop_counts)}, avg={np.mean(hop_counts):.1f}")
    print(f"    Latencies (ms): min={min(latencies):.1f}, max={max(latencies):.1f}, avg={np.mean(latencies):.1f}")
    print(f"    Bandwidths (Mbps): min={min(bandwidths):.1f}, max={max(bandwidths):.1f}, avg={np.mean(bandwidths):.1f}")
    
    # Save selected pair
    selection = {
        'source_as': int(src_as),
        'destination_as': int(dst_as),
        'num_paths': best_count,
        'path_metrics': {
            'hop_counts': hop_counts,
            'latencies': latencies,
            'bandwidths': bandwidths
        }
    }
    
    selection_file = os.path.join(run_dir, "selected_pair.json")
    with open(selection_file, 'w') as f:
        json.dump(selection, f, indent=2)
    print(f"\nSelected pair saved to: {selection_file}")
    
else:
    print("\nWarning: No AS pairs with 5+ paths found!")
    print("Selecting pair with most paths...")
    
    best_pair = max(path_counts.items(), key=lambda x: x[1])
    (src_as, dst_as), count = best_pair
    
    selection = {
        'source_as': int(src_as),
        'destination_as': int(dst_as),
        'num_paths': count
    }
    
    selection_file = os.path.join(run_dir, "selected_pair.json")
    with open(selection_file, 'w') as f:
        json.dump(selection, f, indent=2)

# Create summary statistics
stats = {
    'total_as_pairs': len(path_counts),
    'pairs_with_paths': sum(1 for c in path_counts.values() if c > 0),
    'pairs_with_5plus_paths': len(diverse_pairs),
    'max_paths_for_pair': max(path_counts.values()) if path_counts else 0,
    'avg_paths_per_pair': np.mean(list(path_counts.values())) if path_counts else 0,
    'path_distribution': dict(Counter(path_counts.values()))
}

stats_file = os.path.join(run_dir, "beaconing_stats.json")
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"\nBeaconing statistics saved to: {stats_file}")
print("\nBeaconing simulation complete!")