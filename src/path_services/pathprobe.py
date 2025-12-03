"""
PathProbe service for querying path metrics

Provides fast metric lookups with optional noise injection.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import pandas as pd
# from numba import njit  # Optional optimization


@dataclass
class PathMetrics:
    """Path metrics at a specific time"""
    latency_ms: float
    bandwidth_mbps: float  
    loss_rate: float
    
    def to_dict(self):
        return {
            'latency_ms': self.latency_ms,
            'bandwidth_mbps': self.bandwidth_mbps,
            'loss_rate': self.loss_rate
        }


class PathProbe:
    """Probe path metrics with optional noise"""
    
    def __init__(self,
                 link_metrics_path: Path,
                 link_table_path: Path,
                 metrics_shape: Tuple[int, int, int]):
        """
        Args:
            link_metrics_path: Path to link metrics memmap
            link_table_path: Path to link table parquet
            metrics_shape: Shape of metrics array (T, E, 3)
        """
        # Load link metrics as memory-mapped array
        self.link_metrics = np.memmap(
            link_metrics_path,
            dtype='float32',
            mode='r',
            shape=metrics_shape
        )
        
        # Load link table for edge indexing
        self.link_table = pd.read_pickle(link_table_path)
        
        # Build edge to index mapping
        self.edge_to_idx = {}
        for idx, row in self.link_table.iterrows():
            self.edge_to_idx[(row['u'], row['v'])] = idx
            
        # Noise parameters
        self.latency_noise_std = 2.0  # ms
        self.bandwidth_noise_cv = 0.02  # 2% coefficient of variation
        
    def probe(self, path: 'Path', t_idx: int, noisy: bool = True) -> PathMetrics:
        """
        Query metrics for a path at time t
        
        Args:
            path: Path object from PathFinder
            t_idx: Time slot index
            noisy: Whether to add measurement noise
            
        Returns:
            PathMetrics object
        """
        # Get edge indices for path
        edge_indices = self._get_edge_indices(path)
        
        if not edge_indices:
            # Invalid path
            return PathMetrics(
                latency_ms=float('inf'),
                bandwidth_mbps=0.0,
                loss_rate=1.0
            )
        
        # Extract metrics slice
        metrics_slice = self.link_metrics[t_idx, edge_indices]
        
        # Aggregate metrics
        latency = self._aggregate_latency(metrics_slice[:, 0])
        bandwidth = self._aggregate_bandwidth(metrics_slice[:, 1])
        loss = self._aggregate_loss(metrics_slice[:, 2])
        
        # Add noise if requested
        if noisy:
            latency = self._add_latency_noise(latency)
            bandwidth = self._add_bandwidth_noise(bandwidth)
            
        return PathMetrics(
            latency_ms=latency,
            bandwidth_mbps=bandwidth,
            loss_rate=loss
        )
    
    def probe_batch(self, paths: list, t_idx: int, noisy: bool = True) -> list:
        """
        Probe multiple paths efficiently
        
        Args:
            paths: List of Path objects
            t_idx: Time slot index
            noisy: Whether to add noise
            
        Returns:
            List of PathMetrics
        """
        results = []
        
        # Extract all edge indices first
        all_indices = []
        path_lengths = []
        
        for path in paths:
            indices = self._get_edge_indices(path)
            all_indices.extend(indices)
            path_lengths.append(len(indices))
            
        if not all_indices:
            return [PathMetrics(float('inf'), 0.0, 1.0) for _ in paths]
        
        # Single slice extraction
        all_metrics = self.link_metrics[t_idx, all_indices]
        
        # Process per path
        offset = 0
        for length in path_lengths:
            if length == 0:
                results.append(PathMetrics(float('inf'), 0.0, 1.0))
            else:
                path_metrics = all_metrics[offset:offset+length]
                
                latency = self._aggregate_latency(path_metrics[:, 0])
                bandwidth = self._aggregate_bandwidth(path_metrics[:, 1])
                loss = self._aggregate_loss(path_metrics[:, 2])
                
                if noisy:
                    latency = self._add_latency_noise(latency)
                    bandwidth = self._add_bandwidth_noise(bandwidth)
                    
                results.append(PathMetrics(latency, bandwidth, loss))
                
            offset += length
            
        return results
    
    def _get_edge_indices(self, path: 'Path') -> list:
        """Get link table indices for path edges"""
        indices = []
        
        for i in range(len(path.hops) - 1):
            u, v = path.hops[i], path.hops[i + 1]
            if (u, v) in self.edge_to_idx:
                indices.append(self.edge_to_idx[(u, v)])
                
        return indices
    
    @staticmethod
    def _aggregate_latency(latencies: np.ndarray) -> float:
        """Sum of latencies along path"""
        return np.sum(latencies)
    
    @staticmethod
    def _aggregate_bandwidth(bandwidths: np.ndarray) -> float:
        """Minimum bandwidth along path"""
        return np.min(bandwidths) if len(bandwidths) > 0 else 0.0
    
    @staticmethod
    def _aggregate_loss(losses: np.ndarray) -> float:
        """Combined loss rate: 1 - ∏(1 - loss_i)"""
        no_loss_prob = 1.0
        for loss in losses:
            no_loss_prob *= (1.0 - loss)
        return 1.0 - no_loss_prob
    
    def _add_latency_noise(self, latency: float) -> float:
        """Add Gaussian noise to latency"""
        noise = np.random.normal(0, self.latency_noise_std)
        return max(0.0, latency + noise)
    
    def _add_bandwidth_noise(self, bandwidth: float) -> float:
        """Add multiplicative noise to bandwidth"""
        noise_factor = 1.0 - abs(np.random.normal(0, self.bandwidth_noise_cv))
        return max(0.0, bandwidth * noise_factor)
    
    def get_time_series(self, path: 'Path', 
                       start_t: int, end_t: int,
                       noisy: bool = False) -> dict:
        """
        Get metrics time series for a path
        
        Args:
            path: Path to probe
            start_t: Start time index
            end_t: End time index (exclusive)
            noisy: Whether to add noise
            
        Returns:
            Dictionary with time series arrays
        """
        edge_indices = self._get_edge_indices(path)
        
        if not edge_indices:
            return {
                'latency_ms': np.full(end_t - start_t, float('inf')),
                'bandwidth_mbps': np.zeros(end_t - start_t),
                'loss_rate': np.ones(end_t - start_t)
            }
        
        # Extract time slice
        metrics_slice = self.link_metrics[start_t:end_t, edge_indices]
        
        # Aggregate over time
        latencies = np.array([
            self._aggregate_latency(metrics_slice[t, :, 0])
            for t in range(end_t - start_t)
        ])
        
        bandwidths = np.array([
            self._aggregate_bandwidth(metrics_slice[t, :, 1])
            for t in range(end_t - start_t)
        ])
        
        losses = np.array([
            self._aggregate_loss(metrics_slice[t, :, 2])
            for t in range(end_t - start_t)
        ])
        
        if noisy:
            # Add time-correlated noise
            latency_noise = self._generate_correlated_noise(
                len(latencies), self.latency_noise_std
            )
            latencies = np.maximum(0, latencies + latency_noise)
            
            bw_noise = self._generate_correlated_noise(
                len(bandwidths), self.bandwidth_noise_cv, multiplicative=True
            )
            bandwidths = np.maximum(0, bandwidths * bw_noise)
        
        return {
            'latency_ms': latencies,
            'bandwidth_mbps': bandwidths,
            'loss_rate': losses
        }
    
    def _generate_correlated_noise(self, length: int, std: float,
                                  multiplicative: bool = False) -> np.ndarray:
        """Generate time-correlated noise using AR(1) process"""
        # AR(1) coefficient for temporal correlation
        phi = 0.7
        
        noise = np.zeros(length)
        noise[0] = np.random.normal(0, std)
        
        for t in range(1, length):
            innovation = np.random.normal(0, std * np.sqrt(1 - phi**2))
            noise[t] = phi * noise[t-1] + innovation
            
        if multiplicative:
            # Convert to multiplicative factors centered at 1
            noise = 1.0 + noise
            
        return noise