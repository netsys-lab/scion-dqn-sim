"""
Enhanced state extraction with graph-based and causal features
Based on state-of-the-art research in RL-based path selection
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EnhancedStateExtractor:
    """Enhanced state extractor with graph-based and causal features"""
    
    def __init__(self,
                 max_paths: int = 10,
                 path_features: int = 10,  # Increased from 8
                 network_features: int = 10,  # Increased from 6
                 temporal_features: int = 6,  # Increased from 4
                 graph_features: int = 8,  # New
                 causal_features: int = 6,  # New
                 history_window: int = 10):
        """
        Initialize enhanced state extractor
        
        Args:
            max_paths: Maximum number of paths to consider
            path_features: Features per path
            network_features: Network-wide features
            temporal_features: Time-based features
            graph_features: Graph topology features
            causal_features: Causal relationship features
            history_window: Window for historical features
        """
        self.max_paths = max_paths
        self.path_features = path_features
        self.network_features = network_features
        self.temporal_features = temporal_features
        self.graph_features = graph_features
        self.causal_features = causal_features
        
        # State dimension calculation
        self.state_dim = (
            max_paths * path_features +
            network_features +
            temporal_features +
            graph_features +
            causal_features
        )
        
        # History tracking for causal features
        self.history_window = history_window
        self.congestion_history = deque(maxlen=history_window)
        self.failure_history = deque(maxlen=history_window)
        self.selection_history = deque(maxlen=history_window)
        
        logger.info(f"Enhanced state extractor initialized: state_dim={self.state_dim}")
    
    def extract_state(self,
                     src_as: int,
                     dst_as: int,
                     paths: List[Any],
                     path_metrics: List[Dict],
                     network_state: Dict,
                     network_graph: Optional[Any] = None,
                     current_time: int = 0) -> np.ndarray:
        """
        Extract enhanced state vector
        
        Args:
            src_as: Source AS ID
            dst_as: Destination AS ID
            paths: Available paths
            path_metrics: Metrics for each path
            network_state: Current network state
            network_graph: NetworkX graph of topology
            current_time: Current time in seconds
            
        Returns:
            State vector
        """
        features = []
        
        # 1. Path-specific features (enhanced)
        path_feats = self._extract_enhanced_path_features(
            paths, path_metrics, src_as, dst_as
        )
        features.extend(path_feats)
        
        # 2. Network-wide features (enhanced)
        network_feats = self._extract_enhanced_network_features(
            network_state, len(paths)
        )
        features.extend(network_feats)
        
        # 3. Temporal features (enhanced)
        temporal_feats = self._extract_enhanced_temporal_features(
            current_time, src_as, dst_as
        )
        features.extend(temporal_feats)
        
        # 4. Graph-based features (new)
        if network_graph is not None:
            graph_feats = self._extract_graph_features(
                src_as, dst_as, network_graph, network_state
            )
        else:
            graph_feats = [0.0] * self.graph_features
        features.extend(graph_feats)
        
        # 5. Causal features (new)
        causal_feats = self._extract_causal_features(
            src_as, dst_as, paths, path_metrics
        )
        features.extend(causal_feats)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_enhanced_path_features(self,
                                      paths: List[Any],
                                      metrics: List[Dict],
                                      src_as: int,
                                      dst_as: int) -> List[float]:
        """Extract enhanced features for each path"""
        features = []
        
        for i in range(self.max_paths):
            if i < len(paths) and i < len(metrics):
                path = paths[i]
                metric = metrics[i]
                
                # Basic features (normalized)
                feat = [
                    # 1. Hop count (normalized)
                    self._normalize(len(path.as_sequence), 1, 10),
                    
                    # 2. Path type encoding
                    1.0 if hasattr(path, 'path_type') and 'direct' in path.path_type else 0.0,
                    
                    # 3. Latency (normalized, log scale)
                    self._normalize_log(metric.get('latency_ms', 1000), 1, 1000),
                    
                    # 4. Bandwidth (normalized, log scale)
                    self._normalize_log(metric.get('bandwidth_mbps', 1), 1, 10000),
                    
                    # 5. Loss rate (already 0-1)
                    metric.get('loss_rate', 1.0),
                    
                    # 6. Path reliability score (1 - loss_rate)
                    1.0 - metric.get('loss_rate', 1.0),
                    
                    # 7. Path diversity score (unique hops ratio)
                    len(set(path.as_sequence)) / len(path.as_sequence) if len(path.as_sequence) > 0 else 0,
                    
                    # 8. Core AS ratio
                    self._get_core_as_ratio(path) if hasattr(path, 'as_sequence') else 0,
                    
                    # 9. Path freshness (based on beaconing)
                    self._get_path_freshness(path) if hasattr(path, 'expiration_time') else 0.5,
                    
                    # 10. Path load estimate
                    self._estimate_path_load(metric)
                ]
                features.extend(feat)
            else:
                # Padding for missing paths
                features.extend([0.0] * self.path_features)
        
        return features
    
    def _extract_enhanced_network_features(self,
                                         network_state: Dict,
                                         num_paths: int) -> List[float]:
        """Extract enhanced network-wide features"""
        return [
            # 1. Average link utilization
            network_state.get('avg_link_utilization', 0.5),
            
            # 2. Maximum link utilization
            network_state.get('max_link_utilization', 0.5),
            
            # 3. Congested links ratio
            network_state.get('congested_links_ratio', 0.0),
            
            # 4. Network load variance
            network_state.get('utilization_variance', 0.1),
            
            # 5. Failed paths ratio
            network_state.get('failed_paths_ratio', 0.0),
            
            # 6. Path availability (normalized)
            self._normalize(num_paths, 0, self.max_paths),
            
            # 7. Network diameter estimate
            self._normalize(network_state.get('avg_path_length', 5), 1, 10),
            
            # 8. Active flows (normalized, log scale)
            self._normalize_log(network_state.get('active_flows', 100), 1, 10000),
            
            # 9. Network entropy (path diversity)
            network_state.get('path_entropy', 0.5),
            
            # 10. Congestion trend (-1 to 1)
            self._calculate_congestion_trend()
        ]
    
    def _extract_enhanced_temporal_features(self,
                                          current_time: int,
                                          src_as: int,
                                          dst_as: int) -> List[float]:
        """Extract enhanced temporal features"""
        hour = (current_time // 3600) % 24
        day_of_week = (current_time // 86400) % 7
        
        return [
            # 1. Hour (normalized, cyclic encoding)
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            
            # 3. Day of week (cyclic encoding)
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            
            # 5. Business hours indicator
            1.0 if 8 <= hour <= 17 and day_of_week < 5 else 0.0,
            
            # 6. Time since last selection for this pair
            self._get_time_since_last_selection(src_as, dst_as, current_time)
        ]
    
    def _extract_graph_features(self,
                              src_as: int,
                              dst_as: int,
                              graph: Any,
                              network_state: Dict) -> List[float]:
        """Extract graph-based topology features"""
        features = []
        
        # 1-2. Source node features
        src_degree = graph.degree(src_as) if graph.has_node(src_as) else 0
        features.append(self._normalize(src_degree, 1, 20))
        
        src_centrality = self._get_node_centrality(graph, src_as)
        features.append(src_centrality)
        
        # 3-4. Destination node features
        dst_degree = graph.degree(dst_as) if graph.has_node(dst_as) else 0
        features.append(self._normalize(dst_degree, 1, 20))
        
        dst_centrality = self._get_node_centrality(graph, dst_as)
        features.append(dst_centrality)
        
        # 5. Common neighbors ratio
        if graph.has_node(src_as) and graph.has_node(dst_as):
            src_neighbors = set(graph.neighbors(src_as))
            dst_neighbors = set(graph.neighbors(dst_as))
            common = len(src_neighbors & dst_neighbors)
            total = len(src_neighbors | dst_neighbors)
            features.append(common / total if total > 0 else 0)
        else:
            features.append(0.0)
        
        # 6. Neighborhood congestion (1-hop)
        src_congestion = self._get_neighborhood_congestion(graph, src_as, network_state)
        features.append(src_congestion)
        
        # 7. Neighborhood congestion (1-hop)
        dst_congestion = self._get_neighborhood_congestion(graph, dst_as, network_state)
        features.append(dst_congestion)
        
        # 8. Graph distance estimate (normalized)
        # Simple heuristic based on degree
        dist_estimate = 2 + (1 / (src_degree + 1)) + (1 / (dst_degree + 1))
        features.append(self._normalize(dist_estimate, 1, 10))
        
        return features
    
    def _extract_causal_features(self,
                               src_as: int,
                               dst_as: int,
                               paths: List[Any],
                               metrics: List[Dict]) -> List[float]:
        """Extract causal relationship features"""
        features = []
        
        # 1. Recent congestion events impact
        recent_congestion = self._get_recent_congestion_impact(src_as, dst_as)
        features.append(recent_congestion)
        
        # 2. Recent failure events impact
        recent_failures = self._get_recent_failure_impact(paths)
        features.append(recent_failures)
        
        # 3. Path selection consistency
        consistency = self._get_selection_consistency(src_as, dst_as)
        features.append(consistency)
        
        # 4. Traffic surge indicator
        surge = self._detect_traffic_surge(metrics)
        features.append(surge)
        
        # 5. Performance trend
        trend = self._get_performance_trend(src_as, dst_as)
        features.append(trend)
        
        # 6. Anomaly score
        anomaly = self._calculate_anomaly_score(metrics)
        features.append(anomaly)
        
        return features
    
    # Helper methods
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to [0, 1]"""
        if max_val <= min_val:
            return 0.5
        return np.clip((value - min_val) / (max_val - min_val), 0, 1)
    
    def _normalize_log(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value using log scale"""
        if value <= 0:
            value = min_val
        log_value = np.log(value)
        log_min = np.log(min_val)
        log_max = np.log(max_val)
        return self._normalize(log_value, log_min, log_max)
    
    def _get_core_as_ratio(self, path) -> float:
        """Calculate ratio of core ASes in path"""
        if not hasattr(path, 'hops'):
            return 0.5
        core_count = sum(1 for hop in path.hops if hasattr(hop, 'is_core') and hop.is_core)
        return core_count / len(path.hops) if path.hops else 0
    
    def _get_path_freshness(self, path) -> float:
        """Get path freshness based on expiration"""
        # Placeholder - would use actual beacon expiration
        return 0.8
    
    def _estimate_path_load(self, metric: Dict) -> float:
        """Estimate current load on path"""
        # Based on bandwidth utilization
        bw = metric.get('bandwidth_mbps', 100)
        # Assume higher bandwidth means lower utilization
        return self._normalize(1000 / (bw + 1), 0, 1)
    
    def _calculate_congestion_trend(self) -> float:
        """Calculate recent congestion trend"""
        if len(self.congestion_history) < 2:
            return 0.0
        # Extract congestion values from history
        recent_values = []
        for event in list(self.congestion_history)[-5:]:
            if isinstance(event, dict) and 'severity' in event:
                recent_values.append(event['severity'])
            elif isinstance(event, (int, float)):
                recent_values.append(event)
        
        if len(recent_values) < 2:
            return 0.0
        
        # Simple linear trend
        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        return np.clip(trend, -1, 1)
    
    def _get_time_since_last_selection(self, src: int, dst: int, current_time: int) -> float:
        """Get normalized time since last selection for this pair"""
        # Check history for this src-dst pair
        for entry in reversed(self.selection_history):
            if entry['src'] == src and entry['dst'] == dst:
                time_diff = current_time - entry['time']
                # Normalize to [0, 1] where 1 is very recent
                return np.exp(-time_diff / 3600)  # Decay over 1 hour
        return 0.0
    
    def _get_node_centrality(self, graph, node: int) -> float:
        """Get normalized node centrality"""
        if not graph.has_node(node):
            return 0.0
        # Simple degree centrality
        return self._normalize(graph.degree(node), 0, graph.number_of_nodes() / 10)
    
    def _get_neighborhood_congestion(self, graph, node: int, network_state: Dict) -> float:
        """Get average congestion in node's neighborhood"""
        if not graph.has_node(node):
            return 0.5
        # Placeholder - would aggregate actual link utilizations
        return network_state.get('avg_link_utilization', 0.5)
    
    def _get_recent_congestion_impact(self, src: int, dst: int) -> float:
        """Get impact of recent congestion events"""
        if not self.congestion_history:
            return 0.0
        # Check if recent congestion affected this src-dst pair
        impact = 0.0
        for event in self.congestion_history:
            if event.get('affected_pairs', {}).get((src, dst), False):
                impact += event['severity'] * np.exp(-event['age'] / 3600)
        return np.clip(impact, 0, 1)
    
    def _get_recent_failure_impact(self, paths: List[Any]) -> float:
        """Get impact of recent failures on available paths"""
        if not self.failure_history:
            return 0.0
        # Placeholder
        return 0.0
    
    def _get_selection_consistency(self, src: int, dst: int) -> float:
        """Get path selection consistency for this pair"""
        # Check how consistent past selections were
        pair_selections = [s for s in self.selection_history 
                          if s['src'] == src and s['dst'] == dst]
        if len(pair_selections) < 2:
            return 0.5
        # Calculate consistency metric
        return 0.7  # Placeholder
    
    def _detect_traffic_surge(self, metrics: List[Dict]) -> float:
        """Detect if there's a traffic surge"""
        if not metrics:
            return 0.0
        # High demand = low available bandwidth
        avg_bw = np.mean([m.get('bandwidth_mbps', 100) for m in metrics])
        return self._normalize(1000 / (avg_bw + 1), 0, 1)
    
    def _get_performance_trend(self, src: int, dst: int) -> float:
        """Get recent performance trend for this pair"""
        # Placeholder
        return 0.0
    
    def _calculate_anomaly_score(self, metrics: List[Dict]) -> float:
        """Calculate anomaly score based on metrics"""
        if not metrics:
            return 0.0
        # Simple anomaly: very high latency or loss
        max_latency = max(m.get('latency_ms', 0) for m in metrics)
        max_loss = max(m.get('loss_rate', 0) for m in metrics)
        anomaly = 0.0
        if max_latency > 500:  # >500ms is anomalous
            anomaly += 0.5
        if max_loss > 0.1:  # >10% loss is anomalous
            anomaly += 0.5
        return anomaly
    
    def update_history(self,
                      src: int,
                      dst: int,
                      selected_path: Any,
                      metrics: Dict,
                      current_time: int):
        """Update history for causal features"""
        # Update selection history
        self.selection_history.append({
            'src': src,
            'dst': dst,
            'path': selected_path,
            'metrics': metrics,
            'time': current_time
        })
        
        # Update congestion history if high utilization
        if metrics.get('bandwidth_mbps', float('inf')) < 100:
            self.congestion_history.append({
                'time': current_time,
                'affected_pairs': {(src, dst): True},
                'severity': 0.7,
                'age': 0
            })
        
        # Age existing events
        for event in self.congestion_history:
            event['age'] = current_time - event['time']