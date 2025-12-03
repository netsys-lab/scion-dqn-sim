# SCION AS-Level Simulation Platform - GraphBLAS Implementation

High-performance SCION network simulator using SuiteSparse GraphBLAS for efficient large-scale simulations (10-1000 ASes).

## Overview

This implementation follows the design in `DOCS_INTERNAL/new_design.md` and provides:

- **Topology Generation**: BRITE integration with SCION-specific annotations
- **Link Annotation**: Capacity and delay assignment with queueing models
- **Traffic Engine**: Gravity model with double-peak diurnal patterns
- **Beacon Simulator**: GraphBLAS-optimized SCION beaconing
- **Path Services**: Fast path enumeration and metric queries
- **Algorithm Harness**: Pluggable framework for path selection algorithms

## Key Features

- **Performance**: Simulates 1000 AS networks for 30 days in <10 minutes on 32-core workstation
- **Memory Efficient**: Uses memory-mapped arrays and sparse matrices
- **Parallel Processing**: Leverages multiple cores throughout pipeline
- **Realistic Models**: Gravity traffic, M/M/1 queueing, fiber propagation delays
- **SCION Accurate**: Implements valley-free routing, ISD structure, beaconing

## Installation

```bash
# Install dependencies
pip install numpy scipy pandas networkx scikit-learn numba typer rich pyyaml

# Optional but recommended for best performance
pip install pygraphblas  # Requires SuiteSparse GraphBLAS

# Install in development mode
pip install -e .
```

## Quick Start

```bash
# Generate a 100-AS topology
python -m src_2.cli generate --n-ases 100

# Run full simulation
python -m src_2.cli simulate

# Benchmark algorithms
python -m src_2.cli benchmark --algorithms shortest_path lowest_latency random

# Clean up
python -m src_2.cli clean --force
```

## Architecture

### Data Flow

```
BRITE Config → BRITE Runner → SCION Converter → Link Annotator
                                     ↓
                              Traffic Engine → Link Metrics
                                     ↓
                             Beacon Simulator → Segment Store
                                     ↓
                              Path Services ← Algorithm Harness
```

### Key Components

1. **Topology Layer** (`topology/`)
   - `brite_cfg_gen.py`: Generate BRITE configurations
   - `brite2scion_converter.py`: Convert to SCION with ISDs, roles, link types
   
2. **Link Annotation** (`link_annotation/`)
   - `capacity_delay_builder.py`: Assign capacities and calculate delays
   
3. **Traffic Engine** (`traffic/`)
   - `traffic_engine.py`: Generate time-varying traffic matrices
   
4. **Beacon Simulator** (`beacon/`)
   - `beacon_sim.py`: GraphBLAS-optimized beaconing
   
5. **Path Services** (`path_services/`)
   - `pathfinder.py`: Enumerate valley-free paths
   - `pathprobe.py`: Query path metrics with noise
   
6. **Algorithm Harness** (`harness/`)
   - `algo_harness.py`: Framework for evaluating algorithms

## Configuration

### Topology Configuration (`config/topo-convert.yml`)
```yaml
isd:
  n_isds: 3
  assignment_method: kmeans
core:
  ratio_per_isd: 0.1
links:
  peer_probability: 0.3
```

### Traffic Configuration (`config/traffic.yml`)
```yaml
model:
  type: gravity
  base_traffic_gbps: 10.0
diurnal:
  pattern: double_peak
  morning_peak_hour: 9
  evening_peak_hour: 19
```

### Main Configuration (`config/sim.yml`)
```yaml
topology:
  sizes: [10, 100, 1000]
experiment:
  num_seeds: 10
  algorithms:
    - shortest_path
    - lowest_latency
```

## Performance Optimizations

1. **GraphBLAS Usage**
   - Boolean matrices for beaconing (1 bit/edge)
   - ISO value optimization for frontier tracking
   - Semiring operations for reachability

2. **Memory Management**
   - Memory-mapped arrays for large matrices
   - Sparse matrix storage for segments
   - Process pooling to avoid GIL

3. **Algorithmic**
   - Power-series iteration for core beaconing
   - Top-k filtering per row
   - Precomputed routing matrices

## Testing

```bash
# Run basic tests (no external dependencies)
python src_2/test_basic.py

# Run with example data
python examples/test_graphblas_impl.py
```

## Extending

### Adding a New Algorithm

```python
from src_2.harness.algo_harness import PathSelectionAlgorithm

class MyAlgorithm(PathSelectionAlgorithm):
    def __init__(self):
        super().__init__("my_algorithm")
        
    def select_path(self, src, dst, paths, metrics, flow):
        # Your logic here
        return best_path
```

### Custom Traffic Model

Extend `TrafficEngine` to implement different traffic patterns:

```python
class MyTrafficModel(TrafficEngine):
    def _generate_gravity_matrix(self, node_df):
        # Custom gravity model
        pass
```

## Troubleshooting

### PyGraphBLAS Installation

If you have issues installing pygraphblas:

1. Install SuiteSparse GraphBLAS first:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgraphblas-dev
   
   # macOS
   brew install suite-sparse
   ```

2. Then install pygraphblas:
   ```bash
   pip install pygraphblas
   ```

3. The code will fall back to SciPy sparse matrices if GraphBLAS is unavailable.

### Memory Issues

For large topologies (>500 ASes):

1. Increase memory limits:
   ```bash
   export GB_MEMPOOL=8GiB
   ```

2. Use fewer parallel workers:
   ```bash
   python -m src_2.cli simulate --n-jobs 4
   ```

## Citation

If you use this simulator in your research, please cite:

```bibtex
@software{snetsim_graphblas,
  title = {SCION Network Simulator - GraphBLAS Implementation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo/snetsim}
}
```