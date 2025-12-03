"""
Lowest latency path selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class LowestLatencySelector:
    """Select path with minimum latency"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select path with minimum latency
        
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
            
        # Get latencies from metrics
        latencies = [m['latency_ms'] for m in metrics]
        
        # Return index of lowest latency path
        return np.argmin(latencies)