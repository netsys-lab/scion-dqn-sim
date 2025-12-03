"""
Shortest path selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class ShortestPathSelector:
    """Select path with minimum hop count"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select path with minimum hop count
        
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
            
        # Get hop counts from paths
        hop_counts = [len(p.as_sequence) for p in paths]
        
        # Return index of shortest path
        return np.argmin(hop_counts)