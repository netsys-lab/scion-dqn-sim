#!/usr/bin/env python3
"""
Example usage of the SCION GraphBLAS simulation platform

This script demonstrates how to:
1. Generate a SCION topology
2. Create visualizations
3. Run simulations
"""

from pathlib import Path
from datetime import datetime

# Import our modules
from src_2.topology.brite_cfg_gen import BRITEConfigGenerator
from src_2.topology.brite2scion_converter import BRITE2SCIONConverter
from src_2.link_annotation.capacity_delay_builder import CapacityDelayBuilder
from src_2.visualization.topology_visualizer import create_topology_report


def generate_topology_example(n_ases=50, n_isds=3):
    """Example: Generate and visualize a SCION topology"""
    
    print(f"=== Generating {n_ases}-AS SCION Topology ===\n")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experiments/{timestamp}_example_n{n_ases}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate BRITE configuration
    print("1. Generating BRITE configuration...")
    config_gen = BRITEConfigGenerator()
    config_path = output_dir / "topology.conf"
    config_gen.generate(config_path, n_nodes=n_ases, seed=42)
    
    # Step 2: Create mock BRITE topology (replace with actual BRITE runner)
    print("2. Creating topology...")
    brite_file = create_mock_brite_topology(output_dir, n_ases)
    
    # Step 3: Convert to SCION
    print("3. Converting to SCION topology...")
    converter = BRITE2SCIONConverter(n_isds=n_isds, core_ratio=0.15)
    topology_path = output_dir / "topology.pkl"
    topology = converter.convert(brite_file, topology_path)
    
    print(f"   ✓ {topology['metadata']['n_nodes']} ASes")
    print(f"   ✓ {topology['metadata']['n_edges']} links")
    print(f"   ✓ {topology['metadata']['n_isds']} ISDs")
    print(f"   ✓ {topology['metadata']['n_core_ases']} core ASes")
    
    # Step 4: Annotate links
    print("4. Annotating links with capacity and delay...")
    builder = CapacityDelayBuilder()
    link_table_path = output_dir / "link_table.parquet"
    link_table = builder.annotate(topology_path, link_table_path)
    print(f"   ✓ Annotated {len(link_table)} directional links")
    
    # Step 5: Create visualizations
    print("5. Creating visualizations...")
    vis_dir = output_dir / "visualizations"
    create_topology_report(topology_path, vis_dir)
    
    print(f"\n✅ Complete! Results in: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Topology: {topology_path}")
    print(f"  - Link table: {link_table_path}")
    print(f"  - Visualizations: {vis_dir}/")
    
    return output_dir


def create_mock_brite_topology(output_dir, n_nodes):
    """Create a mock BRITE topology file"""
    import random
    random.seed(42)
    
    content = f"# BRITE topology\\n# Nodes: {n_nodes}\\n"
    
    # Generate nodes
    for i in range(1, n_nodes + 1):
        x = random.uniform(0, 500)
        y = random.uniform(0, 500)
        degree = random.randint(2, 6)
        content += f"{i}\\t{x:.1f}\\t{y:.1f}\\t{degree}\\t{degree}\\t{i}\\t1\\tE_RT\\tU\\n"
    
    content += "\\n# Edges:\\n"
    
    # Generate edges with preferential attachment
    edge_id = 0
    for i in range(2, n_nodes + 1):
        num_connections = min(random.randint(2, 4), i - 1)
        targets = random.sample(range(1, i), num_connections)
        
        for j in targets:
            bw = random.uniform(10, 100)
            content += f"{edge_id}\\t{j}\\t{i}\\t1\\t0.1\\t{bw:.1f}\\t1\\t1\\tE_RT\\tU\\n"
            edge_id += 1
    
    brite_file = output_dir / "topology.brite"
    with open(brite_file, 'w') as f:
        f.write(content)
        
    return brite_file


if __name__ == "__main__":
    # Example 1: Small topology
    print("Example 1: Small 25-AS topology")
    small_dir = generate_topology_example(n_ases=25, n_isds=2)
    
    print("\\n" + "="*50 + "\\n")
    
    # Example 2: Medium topology
    print("Example 2: Medium 100-AS topology")
    medium_dir = generate_topology_example(n_ases=100, n_isds=4)
    
    print("\\n" + "="*50 + "\\n")
    
    print("Examples completed! Check the 'experiments' directory for results.")
    print("\\nVisualization files include:")
    print("  - topology_overview.png: Complete view with statistics")
    print("  - isd_map.png: ISD membership visualization")
    print("  - core_network.png: Core AS connectivity")
    print("  - connectivity_matrix.png: Full adjacency matrix")
    print("  - topology_stats.txt: Detailed statistics")