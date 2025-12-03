"""
SCION default path selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class SCIONDefaultSelector:
    """SCION's default path selection: shortest then freshest"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select path using SCION default algorithm:
        1. Prefer shortest paths
        2. Among shortest, prefer freshest (lowest latency)
        
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
            
        # First criterion: shortest path
        hop_counts = [len(p.as_sequence) for p in paths]
        min_hops = min(hop_counts)
        
        # Among shortest paths, select the one with lowest latency (freshest)
        candidates = [(i, metrics[i]['latency_ms']) 
                     for i, h in enumerate(hop_counts) if h == min_hops]
        
        if candidates:
            # Sort by latency and return index of best
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return 0