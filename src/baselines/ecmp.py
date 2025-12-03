"""
Equal-Cost Multi-Path (ECMP) selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class ECMPSelector:
    """ECMP path selection with hash-based load balancing"""
    
    def __init__(self):
        self.flow_hash_cache = {}
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select among equal-cost paths using flow hash
        
        Args:
            paths: List of available paths
            metrics: List of path metrics
            flow: Current flow information
            state: Current state (unused)
            
        Returns:
            Index of selected path
        """
        if not paths:
            return 0
            
        # Find shortest paths (equal cost)
        hop_counts = [len(p.as_sequence) for p in paths]
        min_hops = min(hop_counts)
        shortest_indices = [i for i, h in enumerate(hop_counts) if h == min_hops]
        
        # Hash-based selection among equal-cost paths
        flow_key = (flow['src'], flow['dst'])
        if flow_key not in self.flow_hash_cache:
            self.flow_hash_cache[flow_key] = hash(flow_key)
        
        selected_idx = self.flow_hash_cache[flow_key] % len(shortest_indices)
        return shortest_indices[selected_idx]