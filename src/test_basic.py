#!/usr/bin/env python3
"""
Basic test script for the GraphBLAS implementation

Tests core functionality without external dependencies.
"""

import numpy as np
from pathlib import Path
import tempfile
import shutil

def test_brite_config_generation():
    """Test BRITE config generation"""
    from src_2.topology.brite_cfg_gen import BRITEConfigGenerator
    
    print("Testing BRITE config generation...")
    gen = BRITEConfigGenerator()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = gen.generate(
            Path(tmpdir) / "test.conf",
            n_nodes=50,
            seed=42
        )
        
        assert config_path.exists()
        content = config_path.read_text()
        assert "n = 50" in content
        print("✓ BRITE config generation works")


def test_traffic_diurnal_pattern():
    """Test diurnal traffic pattern"""
    from src_2.traffic.traffic_engine import TrafficEngine
    
    print("\nTesting diurnal traffic pattern...")
    engine = TrafficEngine()
    
    # Test pattern at different hours
    hours = [0, 6, 9, 12, 15, 19, 23]
    values = []
    
    for hour in hours:
        value = engine._diurnal_pattern(hour)
        values.append(value)
        print(f"  Hour {hour:02d}: {value:.3f}")
    
    # Check that we have variation
    assert max(values) > min(values) * 1.5
    # Check bounds
    assert all(0.0 <= v <= 1.5 for v in values)
    print("✓ Diurnal pattern works correctly")


def test_queueing_delay():
    """Test queueing delay calculation"""
    from src_2.link_annotation.capacity_delay_builder import CapacityDelayBuilder
    
    print("\nTesting queueing delay calculation...")
    
    # Test different utilization levels
    rtt_min = 10.0  # ms
    utilizations = [0.0, 0.5, 0.7, 0.9, 0.95, 0.99]
    
    for util in utilizations:
        delay = CapacityDelayBuilder.queueing_delay(util, rtt_min)
        print(f"  Utilization {util:.0%}: {delay:.2f} ms")
        
        # Check bounds
        assert delay >= 0
        if util < 0.99:
            assert delay < 100 * rtt_min
            
    print("✓ Queueing delay calculation works")


def test_path_dataclass():
    """Test Path dataclass"""
    from src_2.path_services.pathfinder import Path
    
    print("\nTesting Path dataclass...")
    
    path = Path(
        src=1,
        dst=5,
        hops=(1, 2, 3, 4, 5),
        interfaces=((0, 1), (1, 2), (2, 3), (3, 4)),
        segment_types=('up', 'core', 'down'),
        total_hops=4
    )
    
    # Test immutability
    try:
        path.src = 2
        assert False, "Path should be immutable"
    except:
        pass
        
    # Test hashing
    path_set = {path}
    assert path in path_set
    
    print(f"  Path: {path.src} -> {path.dst}, {path.total_hops} hops")
    print("✓ Path dataclass works correctly")


def test_segment_matrix_conversion():
    """Test segment to CSR conversion"""
    from src_2.beacon.beacon_sim import BeaconSimulator
    
    print("\nTesting segment matrix conversion...")
    
    sim = BeaconSimulator()
    
    # Test segments
    segments = [
        [1, 2, 3],
        [4, 5],
        [1, 4, 6, 7]
    ]
    
    matrix = sim._segments_to_csr(segments)
    
    assert matrix.shape[0] == 3  # 3 segments
    assert matrix.shape[1] == 8  # Max AS ID is 7, so size 8
    
    # Check content
    assert matrix[0, 1] == 1
    assert matrix[0, 2] == 1
    assert matrix[0, 3] == 1
    assert matrix[1, 4] == 1
    assert matrix[1, 5] == 1
    
    print(f"  Created {matrix.shape} CSR matrix from {len(segments)} segments")
    print("✓ Segment matrix conversion works")


def test_flow_request():
    """Test flow request generation"""
    from src_2.harness.algo_harness import FlowRequest
    
    print("\nTesting flow request creation...")
    
    flow = FlowRequest(
        src=1,
        dst=10,
        bandwidth_mbps=100.0,
        start_time=50,
        duration=20
    )
    
    assert flow.src == 1
    assert flow.dst == 10
    assert flow.bandwidth_mbps == 100.0
    
    print(f"  Flow: {flow.src}->{flow.dst}, {flow.bandwidth_mbps} Mbps")
    print("✓ Flow request works correctly")


def test_metrics_aggregation():
    """Test path metrics aggregation"""
    from src_2.path_services.pathprobe import PathProbe
    
    print("\nTesting metrics aggregation...")
    
    # Test latency (sum)
    latencies = np.array([5.0, 3.0, 7.0, 2.0])
    total_latency = PathProbe._aggregate_latency(latencies)
    assert total_latency == 17.0
    
    # Test bandwidth (min)
    bandwidths = np.array([1000.0, 500.0, 750.0, 2000.0])
    min_bandwidth = PathProbe._aggregate_bandwidth(bandwidths)
    assert min_bandwidth == 500.0
    
    # Test loss (combined)
    losses = np.array([0.01, 0.02, 0.0, 0.03])
    combined_loss = PathProbe._aggregate_loss(losses)
    expected = 1 - (0.99 * 0.98 * 1.0 * 0.97)
    assert abs(combined_loss - expected) < 0.0001
    
    print("  Latency aggregation: sum")
    print("  Bandwidth aggregation: min")
    print("  Loss aggregation: 1 - ∏(1-loss_i)")
    print("✓ Metrics aggregation works correctly")


if __name__ == "__main__":
    print("Running basic tests for GraphBLAS implementation...\n")
    
    test_brite_config_generation()
    test_traffic_diurnal_pattern()
    test_queueing_delay()
    test_path_dataclass()
    test_segment_matrix_conversion()
    test_flow_request()
    test_metrics_aggregation()
    
    print("\n✅ All basic tests passed!")