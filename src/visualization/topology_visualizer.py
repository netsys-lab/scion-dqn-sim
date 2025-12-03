"""
Topology visualization tool

Creates visual representations of SCION topologies with:
- ISD boundaries and labels
- Core AS highlighting
- Link type differentiation
- Geographic layout preservation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import pickle
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from matplotlib.lines import Line2D


class TopologyVisualizer:
    """Create visualizations of SCION topologies"""
    
    # Color schemes
    ISD_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
    CORE_COLOR = '#2C3E50'
    NON_CORE_COLOR = '#95A5A6'
    
    # Link styles
    LINK_STYLES = {
        'core': {'color': '#E74C3C', 'width': 3.0, 'style': '-'},
        'parent-child': {'color': '#3498DB', 'width': 2.0, 'style': '-'},
        'child-parent': {'color': '#3498DB', 'width': 2.0, 'style': '--'},
        'peer': {'color': '#27AE60', 'width': 1.5, 'style': ':'}
    }
    
    def __init__(self, figsize: Tuple[int, int] = (16, 12)):
        """
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-white')
        
    def visualize_topology(self, topology_path: Path, output_path: Path,
                         show_labels: bool = True,
                         show_grid: bool = True):
        """
        Create comprehensive topology visualization
        
        Args:
            topology_path: Path to topology pickle
            output_path: Where to save the image
            show_labels: Whether to show AS labels
            show_grid: Whether to show grid
        """
        # Load topology
        with open(topology_path, 'rb') as f:
            topology = pickle.load(f)
            
        node_df = topology['nodes']
        edge_df = topology['edges']
        
        # Create figure with subplots
        fig = plt.figure(figsize=self.figsize)
        
        # Main topology plot
        ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        self._draw_topology(ax_main, node_df, edge_df, show_labels)
        
        # Statistics plots
        ax_degree = plt.subplot2grid((3, 3), (0, 2))
        self._plot_degree_distribution(ax_degree, node_df)
        
        ax_isd = plt.subplot2grid((3, 3), (1, 2))
        self._plot_isd_statistics(ax_isd, node_df, edge_df)
        
        ax_links = plt.subplot2grid((3, 3), (2, 2))
        self._plot_link_statistics(ax_links, edge_df)
        
        plt.tight_layout()
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create individual plots
        self._create_individual_plots(topology, output_path.parent)
        
    def _draw_topology(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame,
                      show_labels: bool):
        """Draw the main topology visualization"""
        
        # Create NetworkX graph for layout
        G = nx.Graph()
        for _, node in node_df.iterrows():
            G.add_node(node['as_id'], **node.to_dict())
        
        # Add edges (undirected for visualization)
        seen_edges = set()
        for _, edge in edge_df.iterrows():
            edge_key = tuple(sorted([edge['u'], edge['v']]))
            if edge_key not in seen_edges:
                G.add_edge(edge['u'], edge['v'], type=edge['type'])
                seen_edges.add(edge_key)
        
        # Use geographic positions
        pos = {node['as_id']: (node['x'], node['y']) 
               for _, node in node_df.iterrows()}
        
        # Draw ISD regions first
        self._draw_isd_regions(ax, node_df, pos)
        
        # Draw edges by type
        for link_type, style in self.LINK_STYLES.items():
            edges = [(u, v) for u, v, d in G.edges(data=True) 
                     if d.get('type') == link_type]
            
            nx.draw_networkx_edges(
                G, pos, edgelist=edges,
                edge_color=style['color'],
                width=style['width'],
                style=style['style'],
                alpha=0.7,
                ax=ax
            )
        
        # Draw nodes
        core_nodes = node_df[node_df['role'] == 'core']['as_id'].tolist()
        non_core_nodes = node_df[node_df['role'] == 'non-core']['as_id'].tolist()
        
        # Non-core nodes
        nx.draw_networkx_nodes(
            G, pos, nodelist=non_core_nodes,
            node_color=self.NON_CORE_COLOR,
            node_size=200,
            alpha=0.8,
            ax=ax
        )
        
        # Core nodes (larger and different shape)
        nx.draw_networkx_nodes(
            G, pos, nodelist=core_nodes,
            node_color=self.CORE_COLOR,
            node_size=500,
            node_shape='s',  # Square for core
            alpha=0.9,
            ax=ax
        )
        
        # Labels
        if show_labels:
            labels = {n: str(n) for n in G.nodes()}
            nx.draw_networkx_labels(
                G, pos, labels,
                font_size=8,
                font_color='white',
                font_weight='bold',
                ax=ax
            )
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', 
                   markerfacecolor=self.CORE_COLOR, markersize=10,
                   label='Core AS'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=self.NON_CORE_COLOR, markersize=8,
                   label='Non-core AS'),
        ]
        
        for link_type, style in self.LINK_STYLES.items():
            legend_elements.append(
                Line2D([0], [0], color=style['color'], linewidth=style['width'],
                       linestyle=style['style'], label=f'{link_type.capitalize()} link')
            )
            
        ax.legend(handles=legend_elements, loc='upper left', frameon=True)
        
        # Styling
        ax.set_title('SCION Topology Structure', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        
    def _draw_isd_regions(self, ax, node_df: pd.DataFrame, pos: Dict):
        """Draw convex hulls around ISDs"""
        from scipy.spatial import ConvexHull
        
        for isd in sorted(node_df['isd'].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]['as_id'].tolist()
            if len(isd_nodes) < 3:
                continue
                
            # Get positions
            points = np.array([pos[n] for n in isd_nodes])
            
            # Compute convex hull
            try:
                hull = ConvexHull(points)
                
                # Draw hull with transparency
                hull_points = points[hull.vertices]
                color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
                
                # Add padding
                center = hull_points.mean(axis=0)
                hull_points = center + 1.1 * (hull_points - center)
                
                patch = plt.Polygon(
                    hull_points,
                    alpha=0.2,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=2,
                    linestyle='--'
                )
                ax.add_patch(patch)
                
                # Add ISD label
                ax.text(center[0], center[1], f'ISD {isd}',
                       fontsize=20, fontweight='bold',
                       ha='center', va='center',
                       color=color, alpha=0.7)
                       
            except Exception:
                # Skip if hull computation fails
                pass
                
    def _plot_degree_distribution(self, ax, node_df: pd.DataFrame):
        """Plot degree distribution"""
        degrees = node_df['degree'].values
        
        ax.hist(degrees, bins=range(min(degrees), max(degrees) + 2),
                alpha=0.7, color='#3498DB', edgecolor='black')
        
        ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_degree = degrees.mean()
        ax.axvline(mean_degree, color='red', linestyle='--', alpha=0.7,
                  label=f'Mean: {mean_degree:.1f}')
        ax.legend()
        
    def _plot_isd_statistics(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Plot ISD composition"""
        isd_stats = []
        
        for isd in sorted(node_df['isd'].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]
            
            stats = {
                'ISD': isd,
                'Total': len(isd_nodes),
                'Core': len(isd_nodes[isd_nodes['role'] == 'core']),
                'Non-core': len(isd_nodes[isd_nodes['role'] == 'non-core'])
            }
            isd_stats.append(stats)
            
        isd_df = pd.DataFrame(isd_stats)
        
        # Stacked bar chart
        x = np.arange(len(isd_df))
        width = 0.6
        
        ax.bar(x, isd_df['Core'], width, label='Core',
               color=self.CORE_COLOR, alpha=0.8)
        ax.bar(x, isd_df['Non-core'], width, bottom=isd_df['Core'],
               label='Non-core', color=self.NON_CORE_COLOR, alpha=0.8)
        
        ax.set_title('ISD Composition', fontsize=12, fontweight='bold')
        ax.set_xlabel('ISD')
        ax.set_ylabel('Number of ASes')
        ax.set_xticks(x)
        ax.set_xticklabels(isd_df['ISD'])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
    def _plot_link_statistics(self, ax, edge_df: pd.DataFrame):
        """Plot link type distribution"""
        link_counts = edge_df['type'].value_counts()
        
        colors = [self.LINK_STYLES[lt]['color'] for lt in link_counts.index]
        
        wedges, texts, autotexts = ax.pie(
            link_counts.values,
            labels=link_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax.set_title('Link Type Distribution', fontsize=12, fontweight='bold')
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
    def _create_individual_plots(self, topology: Dict, output_dir: Path):
        """Create additional individual visualizations"""
        node_df = topology['nodes']
        edge_df = topology['edges']
        
        # 1. ISD Map
        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_isd_map(ax, node_df)
        plt.savefig(output_dir / 'isd_map.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Core Network
        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_core_network(ax, node_df, edge_df)
        plt.savefig(output_dir / 'core_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Connectivity Matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        self._create_connectivity_matrix(ax, node_df, edge_df)
        plt.savefig(output_dir / 'connectivity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_isd_map(self, ax, node_df: pd.DataFrame):
        """Create ISD membership map"""
        # Scatter plot colored by ISD
        for isd in sorted(node_df['isd'].unique()):
            isd_nodes = node_df[node_df['isd'] == isd]
            color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
            
            # Plot non-core nodes
            non_core = isd_nodes[isd_nodes['role'] == 'non-core']
            ax.scatter(non_core['x'], non_core['y'], 
                      c=color, s=100, alpha=0.6,
                      label=f'ISD {isd}')
            
            # Plot core nodes
            core = isd_nodes[isd_nodes['role'] == 'core']
            ax.scatter(core['x'], core['y'],
                      c=color, s=300, marker='s',
                      edgecolors='black', linewidths=2)
                      
        ax.set_title('ISD Membership Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _create_core_network(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Visualize only the core network"""
        core_nodes = node_df[node_df['role'] == 'core']['as_id'].tolist()
        core_edges = edge_df[edge_df['type'] == 'core']
        
        # Create core graph
        G = nx.Graph()
        for node in core_nodes:
            node_data = node_df[node_df['as_id'] == node].iloc[0]
            G.add_node(node, isd=node_data['isd'])
            
        for _, edge in core_edges.iterrows():
            if edge['u'] in core_nodes and edge['v'] in core_nodes:
                G.add_edge(edge['u'], edge['v'])
                
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes colored by ISD
        for isd in sorted(node_df['isd'].unique()):
            isd_core_nodes = [n for n in G.nodes() 
                             if G.nodes[n]['isd'] == isd]
            if isd_core_nodes:
                color = self.ISD_COLORS[isd % len(self.ISD_COLORS)]
                nx.draw_networkx_nodes(
                    G, pos, nodelist=isd_core_nodes,
                    node_color=color, node_size=1000,
                    node_shape='s', alpha=0.8, ax=ax
                )
                
        # Draw edges
        nx.draw_networkx_edges(
            G, pos, edge_color=self.LINK_STYLES['core']['color'],
            width=3, alpha=0.7, ax=ax
        )
        
        # Labels
        nx.draw_networkx_labels(
            G, pos, font_size=12, font_color='white',
            font_weight='bold', ax=ax
        )
        
        ax.set_title('Core AS Network', fontsize=16, fontweight='bold')
        ax.axis('off')
        
    def _create_connectivity_matrix(self, ax, node_df: pd.DataFrame, edge_df: pd.DataFrame):
        """Create adjacency matrix visualization"""
        n_nodes = len(node_df)
        matrix = np.zeros((n_nodes, n_nodes))
        
        # Create AS ID to index mapping
        as_to_idx = {row['as_id']: i for i, row in node_df.iterrows()}
        
        # Fill matrix
        for _, edge in edge_df.iterrows():
            i = as_to_idx.get(edge['u'], -1)
            j = as_to_idx.get(edge['v'], -1)
            if i >= 0 and j >= 0:
                # Color code by link type
                link_value = {
                    'core': 4,
                    'parent-child': 3,
                    'child-parent': 2,
                    'peer': 1
                }.get(edge['type'], 0)
                
                matrix[i, j] = link_value
                matrix[j, i] = link_value  # Symmetric
                
        # Plot
        im = ax.imshow(matrix, cmap='YlOrRd', interpolation='nearest')
        
        # Add colorbar with labels
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['None', 'Peer', 'Child-Parent', 'Parent-Child', 'Core'])
        
        ax.set_title('Connectivity Matrix', fontsize=16, fontweight='bold')
        ax.set_xlabel('AS ID Index')
        ax.set_ylabel('AS ID Index')
        
        # Add ISD boundaries
        isd_boundaries = []
        current_idx = 0
        for isd in sorted(node_df['isd'].unique()):
            isd_size = len(node_df[node_df['isd'] == isd])
            isd_boundaries.append(current_idx + isd_size)
            current_idx += isd_size
            
        for boundary in isd_boundaries[:-1]:
            ax.axhline(boundary - 0.5, color='black', linewidth=2)
            ax.axvline(boundary - 0.5, color='black', linewidth=2)


def create_topology_report(topology_path: Path, output_dir: Path):
    """
    Create a comprehensive topology report with visualizations and statistics
    
    Args:
        topology_path: Path to topology pickle
        output_dir: Directory for output files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load topology
    with open(topology_path, 'rb') as f:
        topology = pickle.load(f)
        
    # Create visualizations
    visualizer = TopologyVisualizer()
    visualizer.visualize_topology(
        topology_path,
        output_dir / 'topology_overview.png'
    )
    
    # Generate statistics report
    stats_report = generate_topology_stats(topology)
    
    with open(output_dir / 'topology_stats.txt', 'w') as f:
        f.write(stats_report)
        
    print(f"Topology report generated in {output_dir}")
    

def generate_topology_stats(topology: Dict) -> str:
    """Generate detailed statistics report"""
    node_df = topology['nodes']
    edge_df = topology['edges']
    
    report = []
    report.append("=== SCION Topology Statistics Report ===\n")
    
    # Basic stats
    report.append(f"Total ASes: {len(node_df)}")
    report.append(f"Total Links: {len(edge_df)}")
    report.append(f"Number of ISDs: {len(node_df['isd'].unique())}")
    report.append(f"Core ASes: {len(node_df[node_df['role'] == 'core'])}")
    report.append(f"Average Degree: {node_df['degree'].mean():.2f}")
    
    # ISD details
    report.append("\n=== ISD Breakdown ===")
    for isd in sorted(node_df['isd'].unique()):
        isd_nodes = node_df[node_df['isd'] == isd]
        n_core = len(isd_nodes[isd_nodes['role'] == 'core'])
        n_total = len(isd_nodes)
        report.append(f"ISD {isd}: {n_total} ASes ({n_core} core, {n_total-n_core} non-core)")
        
    # Link analysis
    report.append("\n=== Link Analysis ===")
    link_counts = edge_df['type'].value_counts()
    for link_type, count in link_counts.items():
        percentage = (count / len(edge_df)) * 100
        report.append(f"{link_type}: {count} ({percentage:.1f}%)")
        
    # Connectivity
    report.append("\n=== Connectivity Metrics ===")
    
    # Check if graph is connected
    import networkx as nx
    G = nx.Graph()
    for _, edge in edge_df.iterrows():
        G.add_edge(edge['u'], edge['v'])
        
    if nx.is_connected(G):
        report.append("✓ Topology is fully connected")
        diameter = nx.diameter(G)
        report.append(f"Network diameter: {diameter}")
    else:
        components = list(nx.connected_components(G))
        report.append(f"⚠ Topology has {len(components)} connected components")
        
    return '\n'.join(report)