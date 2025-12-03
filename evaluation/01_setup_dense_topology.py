#!/usr/bin/env python3
"""
Setup dense topology for DQN evaluation with many diverse paths.
Based on evaluation_selective_probing/01_setup_dense_topology.py
"""

import os
import sys
import pickle
import json
import numpy as np
from datetime import datetime
import networkx as nx
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_dense_mock_topology(n_ases=50):
    """Create a dense SCION-like topology with many paths"""
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Node types - more balanced for density
    n_core = 100  # 5 core ASes
    n_tier1 = 300  # 15 tier-1 ASes
    n_tier2 = n_ases - n_core - n_tier1  # 30 tier-2 ASes
    
    # Add nodes with attributes
    node_id = 0
    
    # Core ASes
    core_ases = []
    for i in range(n_core):
        G.add_node(node_id, type='core', name=f'Core-{i}', isd=0)
        core_ases.append(node_id)
        node_id += 1
    
    # Tier-1 ASes
    tier1_ases = []
    for i in range(n_tier1):
        G.add_node(node_id, type='tier1', name=f'Tier1-{i}', isd=0)
        tier1_ases.append(node_id)
        node_id += 1
    
    # Tier-2 ASes
    tier2_ases = []
    for i in range(n_tier2):
        G.add_node(node_id, type='tier2', name=f'Tier2-{i}', isd=0)
        tier2_ases.append(node_id)
        node_id += 1
    
    # Create links with high density
    links = []
    
    # Core full mesh
    for i in range(len(core_ases)):
        for j in range(i+1, len(core_ases)):
            G.add_edge(core_ases[i], core_ases[j], 
                      type='core',
                      bandwidth=10000,  # 10 Gbps
                      latency=np.random.uniform(5, 15))
            links.append({
                'from': core_ases[i],
                'to': core_ases[j],
                'type': 'core',
                'bandwidth': 10000,
                'latency': G[core_ases[i]][core_ases[j]]['latency']
            })
    
    # Tier-1 to Core - INCREASED CONNECTIVITY
    for tier1 in tier1_ases:
        # Connect to 3-4 core ASes
        n_connections = np.random.randint(3, min(5, len(core_ases) + 1))
        connected_cores = np.random.choice(core_ases, n_connections, replace=False)
        
        for core in connected_cores:
            bw = np.random.uniform(2000, 5000)
            lat = np.random.uniform(10, 25)
            G.add_edge(tier1, core,
                      type='customer-provider',
                      bandwidth=bw,
                      latency=lat)
            links.append({
                'from': tier1,
                'to': int(core),
                'type': 'customer-provider',
                'bandwidth': bw,
                'latency': lat
            })
    
    # Tier-1 peering - DENSE MESH
    for i in range(len(tier1_ases)):
        for j in range(i+1, len(tier1_ases)):
            # 40% chance of peering
            if np.random.random() < 0.4:
                bw = np.random.uniform(1000, 3000)
                lat = np.random.uniform(15, 35)
                G.add_edge(tier1_ases[i], tier1_ases[j],
                          type='peer',
                          bandwidth=bw,
                          latency=lat)
                links.append({
                    'from': tier1_ases[i],
                    'to': tier1_ases[j],
                    'type': 'peer',
                    'bandwidth': bw,
                    'latency': lat
                })
    
    # Tier-2 to Tier-1 - MULTIPLE PROVIDERS
    for tier2 in tier2_ases:
        # Connect to 2-4 tier-1 ASes
        n_connections = np.random.randint(2, 5)
        connected_tier1 = np.random.choice(tier1_ases, 
                                         min(n_connections, len(tier1_ases)), 
                                         replace=False)
        
        for tier1 in connected_tier1:
            bw = np.random.uniform(100, 1000)
            lat = np.random.uniform(20, 40)
            G.add_edge(tier2, tier1,
                      type='customer-provider',
                      bandwidth=bw,
                      latency=lat)
            links.append({
                'from': tier2,
                'to': int(tier1),
                'type': 'customer-provider',
                'bandwidth': bw,
                'latency': lat
            })
    
    # Some Tier-2 peering
    for i in range(len(tier2_ases)):
        # Only peer with nearby tier2s
        for j in range(max(0, i-3), min(len(tier2_ases), i+4)):
            if i != j and np.random.random() < 0.1:  # 10% chance
                bw = np.random.uniform(50, 500)
                lat = np.random.uniform(25, 50)
                G.add_edge(tier2_ases[i], tier2_ases[j],
                          type='peer',
                          bandwidth=bw,
                          latency=lat)
    
    topology = {
        'graph': G,
        'links': links
    }
    
    as_categories = {
        'core_ases': core_ases,
        'tier1_ases': tier1_ases,
        'tier2_ases': tier2_ases
    }
    
    return topology, as_categories


def run_dense_beaconing(topology, as_categories):
    """Run beaconing to discover path segments - allow more paths"""
    G = topology['graph']
    core_ases = as_categories['core_ases']
    
    # Store segments
    segment_store = {
        'up': defaultdict(list),
        'down': defaultdict(list),
        'core': defaultdict(list)
    }
    
    # Core segments
    for i in range(len(core_ases)):
        for j in range(len(core_ases)):
            if i != j and G.has_edge(core_ases[i], core_ases[j]):
                segment_store['core'][core_ases[i]].append({
                    'from_as': core_ases[i],
                    'to_as': core_ases[j],
                    'hops': [core_ases[i], core_ases[j]],
                    'total_latency': G[core_ases[i]][core_ases[j]]['latency'],
                    'min_bandwidth': G[core_ases[i]][core_ases[j]]['bandwidth']
                })
    
    # Find paths to core (BFS with longer paths allowed)
    def find_paths_to_core(start_as, max_hops=6):
        paths = []
        queue = [(start_as, [start_as], 0, float('inf'))]
        visited = set()
        
        while queue:
            current, path, total_latency, min_bw = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if we reached a core AS
            if current in core_ases and current != start_as:
                paths.append({
                    'core_as': current,
                    'hops': path,
                    'total_latency': total_latency,
                    'min_bandwidth': min_bw
                })
                # Don't continue from core
                continue
            
            # Explore neighbors
            if len(path) < max_hops:
                for neighbor in G.neighbors(current):
                    if neighbor not in visited or neighbor in core_ases:
                        edge_data = G[current][neighbor]
                        new_latency = total_latency + edge_data['latency']
                        new_bw = min(min_bw, edge_data['bandwidth'])
                        queue.append((neighbor, path + [neighbor], new_latency, new_bw))
        
        return paths
    
    # Generate up segments
    for as_id in G.nodes():
        if as_id not in core_ases:
            paths = find_paths_to_core(as_id)
            # Keep top paths per core
            paths.sort(key=lambda p: (p['total_latency'], -p['min_bandwidth']))
            
            # Group by core and keep best per core
            by_core = defaultdict(list)
            for p in paths:
                by_core[p['core_as']].append(p)
            
            for core, core_paths in by_core.items():
                for path_info in core_paths[:2]:  # Keep 2 best per core
                    segment_store['up'][as_id].append(path_info)
                    
                    # Also create corresponding down segment
                    rev_hops = path_info['hops'][::-1]
                    down_segment = {
                        'core_as': path_info['core_as'],
                        'hops': rev_hops,
                        'total_latency': path_info['total_latency'],
                        'min_bandwidth': path_info['min_bandwidth']
                    }
                    segment_store['down'][as_id].append(down_segment)
    
    # Count segments
    total_segments = (
        sum(len(segs) for segs in segment_store['up'].values()) +
        sum(len(segs) for segs in segment_store['down'].values()) +
        sum(len(segs) for segs in segment_store['core'].values())
    )
    
    print(f"\nDense Beaconing complete:")
    print(f"  - Total segments: {total_segments}")
    print(f"  - Up segments: {sum(len(s) for s in segment_store['up'].values())}")
    print(f"  - Down segments: {sum(len(s) for s in segment_store['down'].values())}")
    print(f"  - Core segments: {sum(len(s) for s in segment_store['core'].values())}")
    
    return segment_store


def select_diverse_src_dst_pair(topology, as_categories, segment_store):
    """Select a source-destination pair with many diverse paths"""
    
    G = topology['graph']
    tier2_ases = as_categories['tier2_ases']
    
    # Try multiple pairs and find one with most paths
    best_pair = None
    max_paths = 0
    best_paths = []
    
    # Sample pairs
    for _ in range(1000):
        src = np.random.choice(tier2_ases)
        dst = np.random.choice(tier2_ases)
        
        if src == dst:
            continue
            
        # Check if they have different providers for diversity
        src_providers = [n for n in G.neighbors(src) if G.nodes[n]['type'] == 'tier1']
        dst_providers = [n for n in G.neighbors(dst) if G.nodes[n]['type'] == 'tier1']
        
        # Good if they have different providers
        if len(set(src_providers) & set(dst_providers)) < len(src_providers):
            # Find paths
            paths = find_available_paths(src, dst, segment_store, max_paths=200)
            if len(paths) > max_paths:
                max_paths = len(paths)
                best_pair = (src, dst)
                best_paths = paths
    
    if best_pair is None:
        # Fallback
        src = tier2_ases[0]
        dst = tier2_ases[-1]
        best_paths = find_available_paths(src, dst, segment_store)
        best_pair = (src, dst)
    
    print(f"\nSelected pair with {len(best_paths)} paths:")
    print(f"  - Source: AS {best_pair[0]}")
    print(f"  - Destination: AS {best_pair[1]}")
    
    return best_pair[0], best_pair[1], best_paths


def find_available_paths(src_as, dst_as, segment_store, max_paths=30):
    """Find all available paths between source and destination"""
    paths = []
    
    # Get up segments from source
    up_segments = segment_store.get('up', {}).get(src_as, [])
    
    # Get down segments to destination
    down_segments = segment_store.get('down', {}).get(dst_as, [])
    
    # Find matching paths through single core
    for up_seg in up_segments:
        for down_seg in down_segments:
            if up_seg['core_as'] == down_seg['core_as']:
                path = {
                    'path_id': f"path_{len(paths)}",
                    'segments': ['up', 'down'],
                    'up_segment': up_seg,
                    'down_segment': down_seg,
                    'core_as': up_seg['core_as'],
                    'hops': up_seg['hops'][:-1] + down_seg['hops'],
                    'static_metrics': {
                        'hop_count': len(up_seg['hops']) + len(down_seg['hops']) - 1,
                        'min_bandwidth': min(up_seg['min_bandwidth'], down_seg['min_bandwidth']),
                        'total_latency': up_seg['total_latency'] + down_seg['total_latency']
                    }
                }
                paths.append(path)
    
    # Also check for paths through multiple core ASes
    core_segments = segment_store.get('core', {})
    for up_seg in up_segments:
        up_core = up_seg['core_as']
        
        if up_core in core_segments:
            for core_seg in core_segments[up_core]:
                other_core = core_seg['to_as']
                
                for down_seg in down_segments:
                    if down_seg['core_as'] == other_core:
                        path = {
                            'path_id': f"path_{len(paths)}",
                            'segments': ['up', 'core', 'down'],
                            'up_segment': up_seg,
                            'core_segment': core_seg,
                            'down_segment': down_seg,
                            'hops': up_seg['hops'][:-1] + core_seg['hops'] + down_seg['hops'][1:],
                            'static_metrics': {
                                'hop_count': len(up_seg['hops']) + len(core_seg['hops']) + len(down_seg['hops']) - 2,
                                'min_bandwidth': min(
                                    up_seg['min_bandwidth'],
                                    core_seg['min_bandwidth'],
                                    down_seg['min_bandwidth']
                                ),
                                'total_latency': (
                                    up_seg['total_latency'] +
                                    core_seg['total_latency'] +
                                    down_seg['total_latency']
                                )
                            }
                        }
                        paths.append(path)
    
    # Sort by hop count and return top paths
    paths.sort(key=lambda p: (p['static_metrics']['hop_count'], p['static_metrics']['total_latency']))
    
    return paths[:max_paths]


def main():
    """Setup dense topology for evaluation"""

    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        run_dir = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        os.makedirs(run_dir, exist_ok=True)
        print(f"Using run directory: {run_dir}")
    
    output_dir = run_dir
    
    # Generate dense topology
    n_ases = 1000
    topology, as_categories = create_dense_mock_topology(n_ases)
    
    print(f"\nDense Topology created:")
    print(f"  - Total ASes: {n_ases}")
    print(f"  - Core ASes: {len(as_categories['core_ases'])}")
    print(f"  - Tier-1 ASes: {len(as_categories['tier1_ases'])}")
    print(f"  - Tier-2 ASes: {len(as_categories['tier2_ases'])}")
    print(f"  - Total links: {topology['graph'].number_of_edges()}")
    
    # Run beaconing
    segment_store = run_dense_beaconing(topology, as_categories)
    
    # Select source-destination pair with many paths
    src_as, dst_as, available_paths = select_diverse_src_dst_pair(topology, as_categories, segment_store)
    
    # Save configuration
    config = {
        'n_ases': n_ases,
        'source_as': int(src_as),
        'destination_as': int(dst_as),
        'n_paths': len(available_paths),
        'as_categories': {
            'core': [int(a) for a in as_categories['core_ases']],
            'tier1': [int(a) for a in as_categories['tier1_ases']],
            'tier2': [int(a) for a in as_categories['tier2_ases']]
        },
        'timestamp': datetime.now().isoformat()
    }
    
    # Save all data
    with open(os.path.join(output_dir, 'dense_topology.pkl'), 'wb') as f:
        pickle.dump(topology, f)
    
    with open(os.path.join(output_dir, 'dense_segment_store.pkl'), 'wb') as f:
        pickle.dump(segment_store, f)
    
    with open(os.path.join(output_dir, 'dense_paths.pkl'), 'wb') as f:
        pickle.dump(available_paths, f)
    
    with open(os.path.join(output_dir, 'dense_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nSaved to {output_dir}/")
    print(f"\nPath diversity:")
    hop_counts = [p['static_metrics']['hop_count'] for p in available_paths]
    print(f"  - Hop counts: {min(hop_counts)} to {max(hop_counts)}")
    print(f"  - Path types: {len([p for p in available_paths if len(p['segments']) == 2])} direct, "
          f"{len([p for p in available_paths if len(p['segments']) == 3])} via core")


if __name__ == "__main__":
    main()