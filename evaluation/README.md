# DQN-based Path Selection Evaluation

This directory contains the complete evaluation pipeline for comparing DQN-based path selection with traditional methods in SCION networks.

## Overview

The evaluation follows the approach described in `simple_dqn.tex`:
- Creates a dense SCION topology
- Runs beaconing to discover paths
- Simulates 28 days of traffic with diurnal patterns
- Trains DQN agent on first 14 days
- Evaluates all methods on last 14 days
- Generates figures

## Key Features

1. **Selective Probing**: DQN only probes selected paths while baseline methods must probe all paths
2. **Differentiated Costs**: Latency probes (10ms) vs bandwidth probes (100ms)
3. **Realistic Traffic**: Diurnal and weekly patterns
4. **Fair Comparison**: All methods evaluated on same traffic flows

## Running the Evaluation

```bash
# Run complete pipeline
python run_full_evaluation.py
or
python run_full_evaluation_2.py

# Or run individual steps with specific run directory
mkdir -p run_YYYYMMDD_HHMMSS
python 01_generate_topology.py run_YYYYMMDD_HHMMSS
python 02_run_beaconing.py run_YYYYMMDD_HHMMSS
python 03_simulate_traffic.py run_YYYYMMDD_HHMMSS
python 04_train_dqn.py run_YYYYMMDD_HHMMSS
python 05_evaluate_methods.py run_YYYYMMDD_HHMMSS
python 06_generate_figures.py run_YYYYMMDD_HHMMSS
```
