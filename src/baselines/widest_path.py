"""
Widest path selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class WidestPathSelector:
    """Select path with maximum bandwidth"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select path with maximum bandwidth
        
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
            
        # Get bandwidths from metrics
        bandwidths = [m['bandwidth_mbps'] for m in metrics]
        
        # Return index of widest path
        return np.argmax(bandwidths)