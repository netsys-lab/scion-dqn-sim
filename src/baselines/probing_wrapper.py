"""
Wrapper for baseline methods to handle probing properly
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ProbingWrapper:
    """
    Wrapper that ensures baseline methods probe all paths before selection
    """
    
    def __init__(self, selector, requires_bandwidth: bool = True):
        """
        Initialize wrapper
        
        Args:
            selector: The baseline selector instance
            requires_bandwidth: Whether this selector needs bandwidth measurements
        """
        self.selector = selector
        self.requires_bandwidth = requires_bandwidth
        self.name = selector.__class__.__name__ if hasattr(selector, '__class__') else 'Unknown'
    
    def select_path_with_probing(self, env) -> int:
        """
        Select path after probing all available paths
        
        Args:
            env: The selective probing environment
            
        Returns:
            Index of selected path
        """
        # Configure probe type based on selector needs
        original_probe_type = env.probe_type
        
        if self.requires_bandwidth:
            env.probe_type = 'full'  # Need bandwidth measurements
        else:
            env.probe_type = 'latency_only'  # Only need latency
        
        # Probe all paths
        metrics = env.probe_all_paths()
        
        # Restore original probe type
        env.probe_type = original_probe_type
        
        # Let baseline selector choose
        action = self.selector.select_path(
            env.available_paths,
            metrics,
            env.current_flow,
            env._get_state()  # Some selectors might use state
        )
        
        return action
    
    def get_probing_requirements(self) -> Dict:
        """Get probing requirements for this selector"""
        return {
            'requires_bandwidth': self.requires_bandwidth,
            'probe_all_paths': True,
            'can_use_historical': False
        }


class AdaptiveBaselineWrapper(ProbingWrapper):
    """
    Advanced wrapper that can use historical data to reduce probing
    """
    
    def __init__(self, selector, requires_bandwidth: bool = True,
                 max_probe_age_seconds: float = 300.0):
        """
        Initialize adaptive wrapper
        
        Args:
            selector: The baseline selector instance
            requires_bandwidth: Whether bandwidth measurements are needed
            max_probe_age_seconds: Maximum age of cached probe data
        """
        super().__init__(selector, requires_bandwidth)
        self.max_probe_age_seconds = max_probe_age_seconds
    
    def select_path_with_adaptive_probing(self, env) -> int:
        """
        Select path with adaptive probing strategy
        
        Args:
            env: The selective probing environment
            
        Returns:
            Index of selected path
        """
        current_time = env.current_time_slot * 900  # Convert to seconds
        paths_to_probe = []
        metrics = []
        
        # Check which paths need fresh probes
        for i, path in enumerate(env.available_paths):
            # Check if we have recent data
            if i in env.probed_path_metrics:
                cached = env.probed_path_metrics[i]
                age = current_time - cached.get('probe_time', 0)
                
                if age < self.max_probe_age_seconds:
                    # Use cached data
                    metrics.append(cached)
                    continue
            
            # Need fresh probe
            paths_to_probe.append(i)
            metrics.append(None)
        
        # Probe only paths that need it
        for idx in paths_to_probe:
            if self.requires_bandwidth:
                probed = env.probe_path_full(idx)
            else:
                probed = env.probe_path_latency(idx)
            
            metrics[idx] = probed
        
        # Fill in any None values (shouldn't happen)
        for i, m in enumerate(metrics):
            if m is None:
                metrics[i] = {
                    'latency_ms': float('inf'),
                    'bandwidth_mbps': 0,
                    'loss_rate': 1.0,
                    'hop_count': 10
                }
        
        # Let baseline selector choose
        action = self.selector.select_path(
            env.available_paths,
            metrics,
            env.current_flow,
            env._get_state()
        )
        
        return action


# Create specific wrappers for each baseline method
class ShortestPathWrapper(ProbingWrapper):
    """Wrapper for shortest path - only needs topology info"""
    def __init__(self, selector):
        # Shortest path doesn't need bandwidth, just hop count
        super().__init__(selector, requires_bandwidth=False)


class WidestPathWrapper(ProbingWrapper):
    """Wrapper for widest path - needs bandwidth measurements"""
    def __init__(self, selector):
        super().__init__(selector, requires_bandwidth=True)


class LowestLatencyWrapper(ProbingWrapper):
    """Wrapper for lowest latency - only needs latency measurements"""
    def __init__(self, selector):
        super().__init__(selector, requires_bandwidth=False)


class ECMPWrapper(ProbingWrapper):
    """Wrapper for ECMP - needs full measurements"""
    def __init__(self, selector):
        super().__init__(selector, requires_bandwidth=True)


class RandomWrapper(ProbingWrapper):
    """Wrapper for random selection - minimal probing"""
    def __init__(self, selector):
        super().__init__(selector, requires_bandwidth=False)
    
    def select_path_with_probing(self, env) -> int:
        """Random doesn't need to probe all paths"""
        # Just probe one random path for consistency
        action = np.random.randint(0, len(env.available_paths))
        env.probe_path_latency(action)
        return action