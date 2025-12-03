"""
Wrapper for baseline path selection methods in fixed-source deployment
Ensures compatibility with academic evaluation methodology
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FixedSourcePathSelector:
    """
    Wrapper for path selection methods in fixed-source deployment
    
    This wrapper ensures that path selection methods work correctly
    when deployed at a specific AS, following academic best practices.
    """
    
    def __init__(self, base_selector, source_as: int, 
                 destination_ases: Optional[List[int]] = None):
        """
        Initialize fixed-source wrapper
        
        Args:
            base_selector: The underlying path selection method
            source_as: The AS where this selector is deployed
            destination_ases: List of valid destination ASes (optional)
        """
        self.base_selector = base_selector
        self.source_as = source_as
        self.destination_ases = destination_ases
        
        # Track performance statistics
        self.selection_stats = {
            'total_selections': 0,
            'destinations_seen': set(),
            'path_diversity': {},
            'selection_consistency': {}
        }
        
        logger.info(f"Initialized {base_selector.__class__.__name__} "
                   f"at AS {source_as}")
    
    def select_path(self, paths: List[Any], metrics: List[Dict], 
                   flow: Dict, state: np.ndarray) -> int:
        """
        Select path ensuring source AS consistency
        
        Args:
            paths: List of available paths
            metrics: List of path metrics
            flow: Current flow information
            state: Current state
            
        Returns:
            Index of selected path
        """
        # Validate source AS
        if flow['src'] != self.source_as:
            logger.warning(f"Flow source {flow['src']} doesn't match "
                          f"deployment AS {self.source_as}")
            # In practice, this shouldn't happen in fixed-source deployment
        
        # Validate destination if configured
        if self.destination_ases and flow['dst'] not in self.destination_ases:
            logger.warning(f"Destination {flow['dst']} not in allowed set")
        
        # Call base selector
        selected_idx = self.base_selector.select_path(paths, metrics, flow, state)
        
        # Track statistics
        self._update_statistics(flow, paths, selected_idx)
        
        return selected_idx
    
    def _update_statistics(self, flow: Dict, paths: List[Any], 
                          selected_idx: int):
        """Update selection statistics for analysis"""
        self.selection_stats['total_selections'] += 1
        self.selection_stats['destinations_seen'].add(flow['dst'])
        
        # Track path diversity per destination
        dst = flow['dst']
        if dst not in self.selection_stats['path_diversity']:
            self.selection_stats['path_diversity'][dst] = set()
        
        if selected_idx < len(paths):
            path_hash = self._hash_path(paths[selected_idx])
            self.selection_stats['path_diversity'][dst].add(path_hash)
        
        # Track selection consistency
        flow_key = (flow['src'], flow['dst'], flow['priority'])
        if flow_key not in self.selection_stats['selection_consistency']:
            self.selection_stats['selection_consistency'][flow_key] = []
        self.selection_stats['selection_consistency'][flow_key].append(selected_idx)
    
    def _hash_path(self, path) -> int:
        """Create hash of path for diversity tracking"""
        return hash(tuple(path.as_sequence))
    
    def get_statistics(self) -> Dict:
        """Get selection statistics"""
        stats = {
            'total_selections': self.selection_stats['total_selections'],
            'unique_destinations': len(self.selection_stats['destinations_seen']),
            'avg_path_diversity': np.mean([
                len(paths) for paths in 
                self.selection_stats['path_diversity'].values()
            ]) if self.selection_stats['path_diversity'] else 0,
            'consistency_score': self._calculate_consistency_score()
        }
        
        return stats
    
    def _calculate_consistency_score(self) -> float:
        """Calculate how consistent selections are for similar flows"""
        if not self.selection_stats['selection_consistency']:
            return 1.0
        
        consistency_scores = []
        
        for flow_key, selections in self.selection_stats['selection_consistency'].items():
            if len(selections) > 1:
                # Calculate fraction of times most common selection was made
                from collections import Counter
                counter = Counter(selections)
                most_common_count = counter.most_common(1)[0][1]
                consistency = most_common_count / len(selections)
                consistency_scores.append(consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def reset_statistics(self):
        """Reset statistics for new evaluation"""
        self.selection_stats = {
            'total_selections': 0,
            'destinations_seen': set(),
            'path_diversity': {},
            'selection_consistency': {}
        }


def create_fixed_source_selector(selector_class, source_as: int,
                               destination_ases: Optional[List[int]] = None,
                               **selector_kwargs):
    """
    Factory function to create fixed-source selector
    
    Args:
        selector_class: The selector class to instantiate
        source_as: Source AS for deployment
        destination_ases: Valid destination ASes
        **selector_kwargs: Additional arguments for selector
        
    Returns:
        FixedSourcePathSelector instance
    """
    base_selector = selector_class(**selector_kwargs)
    return FixedSourcePathSelector(base_selector, source_as, destination_ases)


# Academic evaluation utilities
class PathSelectionEvaluator:
    """
    Evaluator for path selection methods in academic context
    
    Implements proper evaluation methodology for fixed-source deployment
    """
    
    def __init__(self, source_as: int, destination_ases: List[int]):
        """
        Initialize evaluator
        
        Args:
            source_as: Source AS where methods are evaluated
            destination_ases: List of destination ASes
        """
        self.source_as = source_as
        self.destination_ases = destination_ases
        self.evaluation_results = {}
    
    def evaluate_method(self, method_name: str, selector,
                       test_flows: List[Dict], 
                       available_paths_list: List[List],
                       metrics_list: List[List[Dict]],
                       states_list: List[np.ndarray]) -> Dict:
        """
        Evaluate a path selection method
        
        Args:
            method_name: Name of the method
            selector: The path selector (wrapped or unwrapped)
            test_flows: List of test flows
            available_paths_list: Paths for each flow
            metrics_list: Metrics for each flow's paths
            states_list: States for each flow
            
        Returns:
            Evaluation results
        """
        results = {
            'method': method_name,
            'selections': [],
            'latencies': [],
            'bandwidths': [],
            'hop_counts': [],
            'path_diversity': set(),
            'consistency_score': 0.0
        }
        
        # Track selections for consistency
        flow_selections = {}
        
        for i, (flow, paths, metrics, state) in enumerate(
                zip(test_flows, available_paths_list, metrics_list, states_list)):
            
            # Ensure flow uses correct source
            flow['src'] = self.source_as
            
            # Get selection
            selected_idx = selector.select_path(paths, metrics, flow, state)
            
            # Record results
            results['selections'].append(selected_idx)
            
            if selected_idx < len(paths) and selected_idx < len(metrics):
                results['latencies'].append(metrics[selected_idx]['latency_ms'])
                results['bandwidths'].append(metrics[selected_idx]['bandwidth_mbps'])
                results['hop_counts'].append(metrics[selected_idx]['hop_count'])
                
                # Track path diversity
                path_hash = hash(tuple(paths[selected_idx].as_sequence))
                results['path_diversity'].add(path_hash)
                
                # Track for consistency
                flow_key = (flow['dst'], flow['priority'])
                if flow_key not in flow_selections:
                    flow_selections[flow_key] = []
                flow_selections[flow_key].append(selected_idx)
        
        # Calculate consistency
        consistency_scores = []
        for selections in flow_selections.values():
            if len(selections) > 1:
                from collections import Counter
                counter = Counter(selections)
                most_common = counter.most_common(1)[0][1]
                consistency_scores.append(most_common / len(selections))
        
        results['consistency_score'] = (
            np.mean(consistency_scores) if consistency_scores else 1.0
        )
        
        # Calculate summary statistics
        results['summary'] = {
            'avg_latency': np.mean(results['latencies']) if results['latencies'] else np.inf,
            'avg_bandwidth': np.mean(results['bandwidths']) if results['bandwidths'] else 0,
            'avg_hop_count': np.mean(results['hop_counts']) if results['hop_counts'] else 0,
            'path_diversity_count': len(results['path_diversity']),
            'consistency_score': results['consistency_score']
        }
        
        self.evaluation_results[method_name] = results
        
        return results
    
    def compare_methods(self) -> Dict:
        """Compare all evaluated methods"""
        comparison = {
            'rankings': {},
            'summary': {}
        }
        
        # Rank by different metrics
        metrics_to_rank = [
            ('avg_latency', False),  # Lower is better
            ('avg_bandwidth', True),  # Higher is better
            ('path_diversity_count', True),  # Higher is better
            ('consistency_score', True)  # Higher is better
        ]
        
        for metric, higher_better in metrics_to_rank:
            scores = [
                (name, results['summary'][metric])
                for name, results in self.evaluation_results.items()
            ]
            
            # Sort
            scores.sort(key=lambda x: x[1], reverse=higher_better)
            
            comparison['rankings'][metric] = [name for name, _ in scores]
        
        # Summary table
        comparison['summary'] = {
            name: results['summary']
            for name, results in self.evaluation_results.items()
        }
        
        return comparison