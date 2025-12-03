"""
Traffic engine for generating time-varying traffic matrices

Implements gravity model with double-peak diurnal patterns.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
from datetime import datetime, timedelta


class TrafficEngine:
    """Generate time-varying traffic matrices"""
    
    def __init__(self,
                 slot_duration_min: int = 5,
                 horizon_days: int = 30,
                 base_traffic_gbps: float = 10.0):
        """
        Args:
            slot_duration_min: Duration of each time slot in minutes
            horizon_days: Simulation horizon in days
            base_traffic_gbps: Base traffic level in Gbps
        """
        self.slot_duration_min = slot_duration_min
        self.horizon_days = horizon_days
        self.base_traffic_gbps = base_traffic_gbps
        
        # Calculate number of time slots
        self.n_slots = (horizon_days * 24 * 60) // slot_duration_min
        
    def generate(self, topology_path: Path, output_path: Path) -> np.memmap:
        """
        Generate traffic matrices for all time slots
        
        Args:
            topology_path: Path to topology pickle
            output_path: Where to save traffic matrix memmap
            
        Returns:
            Memory-mapped array of shape (T, N, N)
        """
        # Load topology
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
        node_df = topology['nodes']
        n_nodes = len(node_df)
        
        # Generate base gravity matrix
        gravity_matrix = self._generate_gravity_matrix(node_df)
        
        # Create memory-mapped array
        tm_shape = (self.n_slots, n_nodes, n_nodes)
        traffic_matrix = np.memmap(
            output_path,
            dtype='float32',
            mode='w+',
            shape=tm_shape
        )
        
        # Generate time-varying traffic
        for t in range(self.n_slots):
            # Get diurnal factor
            hour = (t * self.slot_duration_min // 60) % 24
            minute = (t * self.slot_duration_min) % 60
            diurnal_factor = self._diurnal_pattern(hour + minute/60)
            
            # Apply to gravity matrix
            traffic_matrix[t] = gravity_matrix * diurnal_factor * self.base_traffic_gbps
            
            # Progress indicator
            if t % 100 == 0:
                print(f"Generated traffic for slot {t}/{self.n_slots}")
        
        # Flush to disk
        del traffic_matrix
        
        return np.memmap(output_path, dtype='float32', mode='r', shape=tm_shape)
    
    def _generate_gravity_matrix(self, node_df: pd.DataFrame) -> np.ndarray:
        """
        Generate gravity model traffic matrix
        
        Traffic between i and j is proportional to:
        (pop_i * pop_j) / distance_ij^2
        """
        n_nodes = len(node_df)
        matrix = np.zeros((n_nodes, n_nodes), dtype='float32')
        
        # Use node degree as proxy for population
        populations = node_df['degree'].values
        positions = node_df[['x', 'y']].values
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i == j:
                    continue
                    
                # Calculate distance
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 1.0:
                    dist = 1.0  # Avoid division by zero
                
                # Gravity model
                traffic = (populations[i] * populations[j]) / (dist ** 2)
                matrix[i, j] = traffic
        
        # Normalize to sum to 1
        total = matrix.sum()
        if total > 0:
            matrix /= total
            
        return matrix
    
    def _diurnal_pattern(self, hour: float) -> float:
        """
        Double-peak diurnal pattern
        
        D(t) = 0.35 + 0.25*sin²(π(t-9)/12) + 0.40*sin²(π(t-19)/12)
        
        Args:
            hour: Hour of day (0-24, can be fractional)
            
        Returns:
            Traffic multiplier (0.35 - 1.0)
        """
        # Morning peak centered at 9 AM
        morning_peak = 0.25 * np.sin(np.pi * (hour - 9) / 12) ** 2
        
        # Evening peak centered at 7 PM (19:00)
        evening_peak = 0.40 * np.sin(np.pi * (hour - 19) / 12) ** 2
        
        # Base traffic + peaks
        return 0.35 + morning_peak + evening_peak


class LinkMetricBuilder:
    """Build link metrics from traffic matrices"""
    
    def __init__(self, n_jobs: int = -1):
        """
        Args:
            n_jobs: Number of parallel jobs for metric computation
        """
        self.n_jobs = n_jobs
        
    def build(self, 
              traffic_path: Path,
              link_table_path: Path, 
              topology_path: Path,
              output_path: Path) -> np.memmap:
        """
        Build time-varying link metrics
        
        Args:
            traffic_path: Path to traffic matrix memmap
            link_table_path: Path to link table parquet
            topology_path: Path to topology pickle
            output_path: Where to save link metrics memmap
            
        Returns:
            Memory-mapped array of shape (T, E, 3) with [latency, residual_bw, loss]
        """
        # Load data
        link_table = pd.read_pickle(link_table_path)
        
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
        n_nodes = topology['metadata']['n_nodes']
        
        # Load traffic matrix shape
        tm_shape = self._get_memmap_shape(traffic_path)
        n_slots = tm_shape[0]
        
        # Build routing matrix (shortest delay paths)
        routing_matrix = self._build_routing_matrix(link_table, n_nodes)
        
        # Create output memmap
        n_edges = len(link_table)
        metrics_shape = (n_slots, n_edges, 3)
        link_metrics = np.memmap(
            output_path,
            dtype='float32',
            mode='w+',
            shape=metrics_shape
        )
        
        # Process each time slot
        traffic_matrix = np.memmap(traffic_path, dtype='float32', mode='r', shape=tm_shape)
        
        from joblib import Parallel, delayed
        
        def process_slot(t):
            # Route traffic
            link_flows = self._route_traffic(traffic_matrix[t], routing_matrix, link_table)
            
            # Calculate metrics
            metrics = self._calculate_link_metrics(link_flows, link_table)
            
            return t, metrics
        
        # Parallel processing
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_slot)(t) for t in range(n_slots)
        )
        
        # Write results
        for t, metrics in results:
            link_metrics[t] = metrics
            if t % 100 == 0:
                print(f"Processed metrics for slot {t}/{n_slots}")
        
        # Flush
        del link_metrics
        
        return np.memmap(output_path, dtype='float32', mode='r', shape=metrics_shape)
    
    def _get_memmap_shape(self, path: Path) -> Tuple[int, int, int]:
        """Infer shape of memory-mapped array"""
        # This is a simplified version - in practice would store metadata
        size = path.stat().st_size
        dtype_size = np.dtype('float32').itemsize
        total_elements = size // dtype_size
        
        # Assume we know it's a 3D array
        # In practice, would save shape metadata separately
        n = int(np.cbrt(total_elements))
        return (total_elements // (n*n), n, n)
    
    def _build_routing_matrix(self, link_table: pd.DataFrame, n_nodes: int) -> Dict:
        """
        Build routing matrix using shortest delay paths
        
        Returns dict mapping (src, dst) -> list of link indices
        """
        # Build adjacency matrix with delays
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra
        
        delays = link_table['prop_delay_ms'].values
        rows = link_table['u'].values
        cols = link_table['v'].values
        
        delay_matrix = csr_matrix((delays, (rows, cols)), shape=(n_nodes, n_nodes))
        
        # Run all-pairs shortest path
        dist_matrix, predecessors = dijkstra(delay_matrix, directed=True, 
                                            return_predecessors=True)
        
        # Build routing table
        routing = {}
        link_index = {(row['u'], row['v']): i for i, row in link_table.iterrows()}
        
        for src in range(n_nodes):
            for dst in range(n_nodes):
                if src == dst:
                    continue
                    
                # Reconstruct path
                path = []
                current = dst
                while current != src and predecessors[src, current] != -9999:
                    prev = predecessors[src, current]
                    if (prev, current) in link_index:
                        path.append(link_index[(prev, current)])
                    current = prev
                
                if path:
                    routing[(src, dst)] = list(reversed(path))
                    
        return routing
    
    def _route_traffic(self, traffic_matrix: np.ndarray, 
                      routing: Dict, link_table: pd.DataFrame) -> np.ndarray:
        """Route traffic matrix over links"""
        link_flows = np.zeros(len(link_table), dtype='float32')
        
        for (src, dst), flow in np.ndenumerate(traffic_matrix):
            if flow > 0 and (src, dst) in routing:
                for link_idx in routing[(src, dst)]:
                    link_flows[link_idx] += flow
                    
        return link_flows
    
    def _calculate_link_metrics(self, link_flows: np.ndarray, 
                               link_table: pd.DataFrame) -> np.ndarray:
        """
        Calculate link metrics: [latency, residual_bw, loss]
        """
        metrics = np.zeros((len(link_table), 3), dtype='float32')
        
        capacities = link_table['capacity_gbps'].values
        prop_delays = link_table['prop_delay_ms'].values
        
        # Calculate utilization
        utilization = link_flows / (capacities * 1000)  # Convert Gbps to Mbps
        utilization = np.clip(utilization, 0, 0.99)
        
        # Latency = propagation + queueing
        from ..link_annotation.capacity_delay_builder import CapacityDelayBuilder
        queueing_delays = np.array([
            CapacityDelayBuilder.queueing_delay(u, prop_delays[i] * 2)  # RTT approx
            for i, u in enumerate(utilization)
        ])
        
        metrics[:, 0] = prop_delays + queueing_delays
        
        # Residual bandwidth (Mbps)
        metrics[:, 1] = capacities * 1000 * (1 - utilization)
        
        # Loss rate (0 if utilization < 0.7, linear increase after)
        loss = np.where(utilization < 0.7, 0, 
                       0.01 * (utilization - 0.7) / 0.3)
        metrics[:, 2] = loss
        
        return metrics