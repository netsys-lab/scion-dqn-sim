# SCION Implementation Notes - src_2

## Overview

This implementation follows the new design from `DOCS_INTERNAL/new_design.md` with corrections based on the SCION control plane specification (draft-dekater-scion-controlplane-09).

## Key Corrections Made

### 1. Interface Tracking
- **Issue**: Original implementation was missing interface IDs in path segments
- **Fix**: All segments now properly track ingress and egress interfaces for each hop
- **Impact**: Paths can now be correctly validated and forwarded in the data plane

### 2. Beacon Propagation
- **Issue**: Matrix-based beaconing lost interface information during propagation
- **Fix**: Implemented PCB-based propagation in `beacon_sim_v2.py` that maintains full hop information
- **Impact**: Correct SCION beaconing behavior with proper segment construction

### 3. Path Structure
- **Issue**: Paths were represented as simple AS sequences
- **Fix**: Implemented `PathHop` dataclass with AS ID and interface information
- **Impact**: Enables proper SCION forwarding and path validation

### 4. Segment Storage
- **Issue**: Segments stored minimal information
- **Fix**: Segments now include:
  - Full hop sequence with interfaces
  - Segment type and ISD information
  - Timestamps and segment IDs
  - Metadata for debugging

## Architecture Components

### Topology Layer (`topology/`)
- `brite_cfg_gen.py`: BRITE configuration generator
- `brite2scion_converter.py`: Converts BRITE to SCION with ISD assignment

### Beacon Simulator (`beacon/`)
- `beacon_sim.py`: Original matrix-based simulator (performance optimized)
- `beacon_sim_v2.py`: Corrected simulator with full interface tracking

### Path Services (`path_services/`)
- `pathfinder_v2.py`: Path enumeration with proper SCION path structure
- `pathprobe.py`: Path metric probing for algorithm evaluation

### Traffic Engine (`traffic/`)
- `traffic_engine.py`: Gravity-based traffic generation
- Supports diurnal patterns and flash crowds

### Algorithm Harness (`harness/`)
- Framework for evaluating path selection algorithms
- Baseline algorithms: shortest path, lowest latency, widest path

### Visualization (`visualization/`)
- Comprehensive topology visualization
- Creates multiple views: ISDs, connectivity, geographic

## Performance Considerations

1. **GraphBLAS Fallback**: When GraphBLAS is not available, falls back to SciPy sparse matrices
2. **Memory-Mapped Arrays**: Used for large metric data to avoid memory issues
3. **Parallel Processing**: Intra-ISD beaconing runs in parallel
4. **Sparse Matrices**: Efficient representation of topology connectivity

## Usage Example

```python
from src_2.beacon.beacon_sim_v2 import CorrectedBeaconSimulator
from src_2.path_services.pathfinder_v2 import PathFinderV2

# Run beacon simulation
simulator = CorrectedBeaconSimulator()
segment_store, stats = simulator.simulate(topology_path, output_dir)

# Find paths
pathfinder = PathFinderV2(topology_path, segment_store_path, link_table_path)
paths = pathfinder.get_paths(src_as, dst_as, k=5, policy="min-lat")
```

## Known Limitations

1. **Sparse Connectivity**: Small topologies (< 100 ASes) may have limited segment coverage
2. **Peering Links**: Not fully utilized in current path construction
3. **Path Registration**: Simplified - doesn't model full SCION path server behavior
4. **Crypto**: Path validation crypto (MACs) not implemented

## Future Improvements

1. Implement full SCION crypto for path validation
2. Add peering shortcuts in path construction
3. Model path server behavior more accurately
4. Add support for hidden paths and path authorization