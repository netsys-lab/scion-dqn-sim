"""
RL agent wrapper that implements selective probing strategy
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SelectiveProbingRLAgent:
    """
    Wrapper for RL agents that implements intelligent selective probing
    """
    
    def __init__(self, base_agent, 
                 exploration_budget: int = 2,
                 probe_strategy: str = 'adaptive',
                 confidence_threshold: float = 0.8):
        """
        Initialize selective probing RL agent
        
        Args:
            base_agent: The underlying DQN agent
            exploration_budget: Number of additional paths to probe for exploration
            probe_strategy: 'minimal', 'exploration', or 'adaptive'
            confidence_threshold: Confidence needed to skip bandwidth probing
        """
        self.base_agent = base_agent
        self.exploration_budget = exploration_budget
        self.probe_strategy = probe_strategy
        self.confidence_threshold = confidence_threshold
        
        # Track selection history for adaptive strategies
        self.selection_history = {}
        self.probe_history = {}
    
    def act_with_selective_probing(self, env, state: np.ndarray, 
                                  valid_actions: List[int]) -> Tuple[int, Dict]:
        """
        Select action with intelligent probing strategy
        
        Args:
            env: The selective probing environment
            state: Current state
            valid_actions: List of valid action indices
            
        Returns:
            Tuple of (selected_action, probing_info)
        """
        # Get action from base agent (without probing)
        action = self.base_agent.act(state, valid_actions)
        
        # Determine which paths to probe
        paths_to_probe = self._determine_paths_to_probe(
            env, action, valid_actions
        )
        
        # Probe selected paths
        probing_info = {
            'paths_probed': paths_to_probe,
            'probe_types': {},
            'num_latency_probes': 0,
            'num_bandwidth_probes': 0
        }
        
        for path_idx in paths_to_probe:
            probe_type = self._determine_probe_type(env, path_idx)
            
            if probe_type == 'full':
                env.probe_path_full(path_idx)
                probing_info['num_latency_probes'] += 1
                probing_info['num_bandwidth_probes'] += 1
            else:  # latency_only
                env.probe_path_latency(path_idx)
                probing_info['num_latency_probes'] += 1
            
            probing_info['probe_types'][path_idx] = probe_type
        
        # Update history
        self._update_history(env, action, paths_to_probe)
        
        return action, probing_info
    
    def _determine_paths_to_probe(self, env, selected_action: int, 
                                 valid_actions: List[int]) -> List[int]:
        """
        Determine which paths to probe based on strategy
        """
        paths_to_probe = []
        
        if self.probe_strategy == 'minimal':
            # Only probe the selected path
            paths_to_probe = [selected_action]
            
        elif self.probe_strategy == 'exploration':
            # Probe selected + exploration paths
            paths_to_probe = [selected_action]
            
            # Add exploration paths
            if self.base_agent.epsilon > 0.1 or hasattr(self.base_agent, 'config') and \
               self.base_agent.config.use_noisy_nets:
                # Still exploring - probe more paths
                other_actions = [a for a in valid_actions if a != selected_action]
                if other_actions:
                    # Prioritize least recently probed
                    exploration_paths = self._get_exploration_priority(
                        env, other_actions
                    )
                    paths_to_probe.extend(exploration_paths[:self.exploration_budget])
                    
        else:  # adaptive
            # Always probe selected path
            paths_to_probe = [selected_action]
            
            # Adaptive exploration based on confidence
            confidence = self._estimate_decision_confidence(env, selected_action)
            
            if confidence < self.confidence_threshold:
                # Low confidence - probe more paths
                other_actions = [a for a in valid_actions if a != selected_action]
                if other_actions:
                    # Get top alternative paths
                    alternatives = self._get_top_alternatives(
                        env, other_actions, n=self.exploration_budget
                    )
                    paths_to_probe.extend(alternatives)
        
        return paths_to_probe
    
    def _determine_probe_type(self, env, path_idx: int) -> str:
        """
        Determine whether to do latency-only or full probe
        """
        # Check if we have recent historical bandwidth data
        hist = env.get_historical_estimate(path_idx)
        
        if hist and hist.get('confidence', 0) > 0.7:
            # High confidence in historical data - latency probe is enough
            return 'latency_only'
        
        # Check if path is selected action
        current_flow_key = (env.current_flow['src'], env.current_flow['dst'])
        if path_idx == 0:  # Assuming this is selected path
            # For selected path, check if flow requires bandwidth guarantee
            if env.current_flow.get('priority') == 'high' or \
               env.current_flow.get('size_mb', 0) > 100:
                # High priority or large flow - need bandwidth probe
                return 'full'
        
        # Default based on probe cost-benefit
        if env.bandwidth_probe_cost_ms > 50:
            # Expensive bandwidth probe - be selective
            return 'latency_only'
        else:
            return 'full'
    
    def _get_exploration_priority(self, env, actions: List[int]) -> List[int]:
        """
        Get exploration priority based on least recently probed
        """
        current_time = env.current_time_slot * 900
        flow_key = (env.current_flow['src'], env.current_flow['dst'])
        
        action_ages = []
        for action in actions:
            if flow_key in self.probe_history and action in self.probe_history[flow_key]:
                age = current_time - self.probe_history[flow_key][action]
            else:
                age = float('inf')  # Never probed
            
            action_ages.append((action, age))
        
        # Sort by age (oldest first)
        action_ages.sort(key=lambda x: x[1], reverse=True)
        
        return [a for a, _ in action_ages]
    
    def _get_top_alternatives(self, env, actions: List[int], n: int) -> List[int]:
        """
        Get top alternative paths based on Q-values
        """
        if not hasattr(self.base_agent, 'q_network'):
            # Fallback to random selection
            return np.random.choice(actions, size=min(n, len(actions)), 
                                  replace=False).tolist()
        
        # Get Q-values for all actions
        if not TORCH_AVAILABLE:
            return np.random.choice(actions, size=min(n, len(actions)), 
                                  replace=False).tolist()
        
        state_tensor = self.base_agent._prepare_state(env._get_state())
        with torch.no_grad():
            q_values = self.base_agent.q_network(state_tensor).cpu().numpy()
        
        # Get indices of top alternatives
        action_values = [(a, q_values[0][a]) for a in actions]
        action_values.sort(key=lambda x: x[1], reverse=True)
        
        return [a for a, _ in action_values[:n]]
    
    def _estimate_decision_confidence(self, env, selected_action: int) -> float:
        """
        Estimate confidence in the selected action
        """
        if not hasattr(self.base_agent, 'q_network'):
            return 0.5  # No confidence info available
        
        # Get Q-values
        if not TORCH_AVAILABLE:
            return 0.5
            
        # Convert state to tensor
        state = env._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.base_agent.device)
        
        # Set to eval mode to avoid batch norm issues
        self.base_agent.q_network.eval()
        
        with torch.no_grad():
            q_values = self.base_agent.q_network(state_tensor).cpu().numpy()[0]
        
        # Set back to train mode
        self.base_agent.q_network.train()
        
        # Calculate confidence based on Q-value separation
        selected_q = q_values[selected_action]
        other_qs = [q_values[i] for i in range(len(q_values)) if i != selected_action]
        
        if not other_qs:
            return 1.0
        
        # Confidence = how much better is selected action
        max_other_q = max(other_qs)
        q_diff = selected_q - max_other_q
        
        # Normalize to [0, 1]
        confidence = np.tanh(q_diff * 2)  # Sigmoid-like mapping
        
        return max(0.0, confidence)
    
    def _update_history(self, env, selected_action: int, probed_paths: List[int]):
        """Update selection and probe history"""
        current_time = env.current_time_slot * 900
        flow_key = (env.current_flow['src'], env.current_flow['dst'])
        
        # Update selection history
        if flow_key not in self.selection_history:
            self.selection_history[flow_key] = {}
        self.selection_history[flow_key][selected_action] = current_time
        
        # Update probe history
        if flow_key not in self.probe_history:
            self.probe_history[flow_key] = {}
        
        for path_idx in probed_paths:
            self.probe_history[flow_key][path_idx] = current_time
    
    def get_probing_requirements(self) -> Dict:
        """Get probing requirements for this RL agent"""
        return {
            'requires_bandwidth': False,  # Can work without bandwidth
            'probe_all_paths': False,     # Selective probing
            'can_use_historical': True,   # Uses historical data
            'exploration_budget': self.exploration_budget,
            'probe_strategy': self.probe_strategy
        }
    
    def train(self, *args, **kwargs):
        """Forward training calls to base agent"""
        return self.base_agent.train(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        """Forward save calls to base agent"""
        return self.base_agent.save(*args, **kwargs)
    
    def load(self, *args, **kwargs):
        """Forward load calls to base agent"""
        return self.base_agent.load(*args, **kwargs)


# Import torch at module level
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - some RL features will be limited")