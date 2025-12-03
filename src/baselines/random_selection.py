"""
Random path selection algorithm
"""

import numpy as np
from typing import List, Dict, Any


class RandomSelector:
    """Randomly select from available paths"""
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Randomly select a path
        
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
            
        return np.random.randint(0, len(paths))