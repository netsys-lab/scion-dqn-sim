#!/usr/bin/env python3
"""
Generate LNCS-style figures showing exploration-aware probing behavior
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter

# Configure matplotlib for LNCS style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 10
rcParams['axes.titlesize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.titlesize'] = 12

# LNCS column width is ~3.5 inches, full width is ~7 inches
COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.0

# Method display names
METHOD_NAMES = {
    'dqn': 'DQN',
    'shortest_path': 'Shortest Path',
    'widest_path': 'Widest Path', 
    'lowest_latency': 'Lowest Latency',
    'ecmp': 'ECMP',
    'random': 'Random'
}

# Method colors
METHOD_COLORS = {
    'dqn': '#e74c3c',  # Red (highlight our method)
    'shortest_path': '#3498db',  # Blue
    'widest_path': '#2ecc71',  # Green
    'lowest_latency': '#f39c12',  # Orange
    'ecmp': '#9b59b6',  # Purple
    'random': '#95a5a6',  # Gray
}


def calculate_effective_probes(results, window_size=1000):
    """Calculate effective probes considering exploration over a window"""
    effective_probes = {}
    
    for method in results:
        if method == 'scion_default':
            continue
            
        if method == 'dqn':
            # For DQN, calculate unique paths probed in sliding windows
            selected_paths = results[method]['raw_results']['selected_paths']
            n_flows = len(selected_paths)
            
            unique_paths_per_window = []
            for i in range(0, n_flows - window_size, window_size // 10):
                unique_in_window = len(set(selected_paths[i:i+window_size]))
                unique_paths_per_window.append(unique_in_window)
            
            # Average unique paths probed per window
            avg_unique = np.mean(unique_paths_per_window)
            # Each path requires 2 probes (latency + bandwidth)
            effective_probes[method] = avg_unique * 2 / window_size * window_size
        else:
            # Baseline methods probe consistently
            total_probes = (results[method]['probing_stats']['latency_probes'] + 
                           results[method]['probing_stats']['bandwidth_probes'])
            n_flows = len(results[method]['raw_results']['rewards'])
            effective_probes[method] = total_probes / n_flows
    
    return effective_probes


def main():
    """Generate LNCS-style figures with exploration-aware metrics"""
    
    output_dir = "evaluation_output"
    
    # Load results
    print("Loading results...")
    with open(os.path.join(output_dir, "evaluation_results_fixed_v2.pkl"), 'rb') as f:
        results = pickle.load(f)
    
    # Remove SCION default if it exists
    if 'scion_default' in results:
        del results['scion_default']
    
    # Get actual number of evaluation flows
    n_flows = len(results[list(results.keys())[0]]['raw_results']['rewards'])
    
    # Figure 1: Probing Behavior (Two perspectives)
    print("\nGenerating Figure 1: Probing behavior...")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.5))
    
    # Order methods for display
    method_order = ['dqn', 'shortest_path', 'lowest_latency', 'widest_path', 'ecmp', 'random']
    
    # Subplot 1: Probes per individual selection (current view)
    probes_per_selection = []
    for m in method_order:
        if m in ['shortest_path', 'random']:
            probes_per_selection.append(0)
        elif m == 'dqn':
            probes_per_selection.append(2)  # Always 2 per selection
        else:
            # Other methods probe all paths
            total_probes = (results[m]['probing_stats']['latency_probes'] + 
                           results[m]['probing_stats']['bandwidth_probes'])
            probes_per_flow = total_probes / n_flows
            probes_per_selection.append(probes_per_flow)
    
    colors = [METHOD_COLORS[m] for m in method_order]
    names = [METHOD_NAMES[m] for m in method_order]
    
    x_pos = np.arange(len(method_order))
    bars1 = ax1.bar(x_pos, probes_per_selection, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight DQN
    bars1[0].set_alpha(1.0)
    bars1[0].set_linewidth(1.5)
    
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Probes per Selection')
    ax1.set_title('(a) Instantaneous Probe Count')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 60)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, probes_per_selection)):
        height = bar.get_height()
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        else:
            ax1.text(bar.get_x() + bar.get_width()/2, 0.5,
                    '0', ha='center', va='bottom', fontsize=8)
    
    # Add note for DQN
    ax1.text(0, 5, 'Per-selection:\nalways 2 probes\n(one path)', 
             ha='center', fontsize=7, color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))
    
    # Subplot 2: Effective probes over 1000-flow window
    window_size = 1000
    effective_probes = []
    
    for m in method_order:
        if m in ['shortest_path', 'random']:
            effective_probes.append(0)
        elif m == 'dqn':
            # Calculate average unique paths in sliding windows
            selected_paths = results[m]['raw_results']['selected_paths']
            unique_counts = []
            for i in range(0, len(selected_paths) - window_size, 100):
                unique_in_window = len(set(selected_paths[i:i+window_size]))
                unique_counts.append(unique_in_window)
            avg_unique_paths = np.mean(unique_counts)
            # 2 probes per path, but spread over window
            effective_probe_rate = avg_unique_paths * 2
            effective_probes.append(effective_probe_rate)
        else:
            # Baseline methods probe all paths every time
            total_probes = (results[m]['probing_stats']['latency_probes'] + 
                           results[m]['probing_stats']['bandwidth_probes'])
            probes_per_flow = total_probes / n_flows
            effective_probes.append(probes_per_flow * window_size)
    
    bars2 = ax2.bar(x_pos, effective_probes, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Highlight DQN
    bars2[0].set_alpha(1.0)
    bars2[0].set_linewidth(1.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel(f'Total Probes per {window_size} Flows')
    ax2.set_title(f'(b) Cumulative Probes Over {window_size} Flows')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_yscale('log')
    ax2.set_ylim(1, 100000)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, effective_probes)):
        if val > 0:
            height = bar.get_height()
            y_pos = height * 1.5
            ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=8)
    
    # Add exploration note for DQN
    dqn_stats = results['dqn']['probing_stats']
    ax2.text(0, 80, f'Explores {len(dqn_stats["explored_paths"])}\nunique paths\n(ε=0.05)', 
             ha='center', fontsize=7, color='darkred',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.7))
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'figure1_probing_exploration.pdf'), dpi=300, bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'figure1_probing_exploration.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print analysis
    print("\n" + "="*80)
    print("PROBING BEHAVIOR ANALYSIS")
    print("="*80)
    
    # DQN exploration analysis
    dqn_paths = results['dqn']['raw_results']['selected_paths']
    path_counts = Counter(dqn_paths)
    
    print(f"\nDQN Exploration Statistics:")
    print(f"  Total selections: {len(dqn_paths)}")
    print(f"  Unique paths explored: {len(dqn_stats['explored_paths'])}/25")
    print(f"  Exploration rate: {dqn_stats['exploration_count'] / len(dqn_paths):.1%}")
    print(f"  Most selected path: {path_counts.most_common(1)[0][0]} ({path_counts.most_common(1)[0][1]/len(dqn_paths):.1%})")
    
    # Window analysis
    for window in [100, 500, 1000, 5000]:
        unique_counts = []
        for i in range(0, len(dqn_paths) - window, window//10):
            unique_counts.append(len(set(dqn_paths[i:i+window])))
        print(f"  Average unique paths per {window} flows: {np.mean(unique_counts):.1f}")
    
    # Probe reduction
    baseline_total = []
    for m in ['widest_path', 'lowest_latency', 'ecmp']:
        total = results[m]['probing_stats']['latency_probes'] + results[m]['probing_stats']['bandwidth_probes']
        baseline_total.append(total)
    
    dqn_total = results['dqn']['probing_stats']['latency_probes'] + results['dqn']['probing_stats']['bandwidth_probes']
    reduction = (np.mean(baseline_total) - dqn_total) / np.mean(baseline_total) * 100
    
    print(f"\n  Total probe reduction: {reduction:.1f}%")
    print(f"  DQN total probes: {dqn_total:,}")
    print(f"  Baseline average: {int(np.mean(baseline_total)):,}")


if __name__ == "__main__":
    main()