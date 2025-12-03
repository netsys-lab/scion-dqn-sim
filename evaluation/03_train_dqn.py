#!/usr/bin/env python3
"""
Train DQN agent following the approach in simple_dqn.tex
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DQN(nn.Module):
    """Deep Q-Network as described in simple_dqn.tex"""
    def __init__(self, state_dim, action_dim, hidden_sizes=[128, 64]):
        super(DQN, self).__init__()
        layers = []
        prev_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class SelectiveProbingEnvironment:
    """Environment that simulates selective probing"""
    def __init__(self, flows, paths):
        self.flows = flows
        self.paths = paths
        self.n_paths = len(paths)
        self.current_flow_idx = 0
        self.probing_stats = {
            'latency_probes': 0,
            'bandwidth_probes': 0,
            'total_probe_time': 0
        }
        
    def reset(self):
        """Reset to a random flow"""
        self.current_flow_idx = random.randint(0, len(self.flows) - 1)
        return self._get_state()
    
    def _get_state(self):
        """Get state features as described in simple_dqn.tex"""
        flow = self.flows[self.current_flow_idx]
        
        # Extract time features
        time_min = flow['start_time']
        hour = (time_min // 60) % 24
        day_of_week = (time_min // (24 * 60)) % 7
        
        # State features from simple_dqn.tex
        state = np.array([
            day_of_week / 6.0,  # Day of week normalized
            hour / 23.0,  # Time of day normalized
            flow['requested_rate_mbps'] / 1000.0,  # Link bandwidth normalized
            0.5,  # Link capacity utilization (placeholder)
            0.7   # Link trust score (placeholder)
        ], dtype=np.float32)
        
        return state
    
    def step(self, action):
        """Take action (select path) and return reward"""
        flow = self.flows[self.current_flow_idx]
        
        # DQN selectively probes only the chosen path
        if action < len(self.paths):
            path = self.paths[action]
            
            # Simulate probing cost
            hop_count = path['static_metrics']['hop_count']
            self.probing_stats['bandwidth_probes'] += 1
            self.probing_stats['total_probe_time'] += 100 + 20 * hop_count
            
            # Get actual performance for this path
            path_perf = next(p for p in flow['path_performance'] if p['path_id'] == path['path_id'])
            
            # Calculate reward as per simple_dqn.tex
            # Goodput normalized to [0, 1] based on max 50 Mbps
            goodput = min(path_perf['bandwidth'], 50) / 50.0
            
            # Link trust calculation
            w3, w4 = 0.5, 0.5
            packet_loss = 1 - path_perf['success_prob']
            delay_normalized = min(path_perf['latency'], 100) / 100.0
            link_trust = 1 - (w3 * packet_loss + w4 * delay_normalized)
            
            # Final reward: r = 2(w1 * G + w2 * T) - 1
            w1, w2 = 0.7, 0.3
            reward = 2 * (w1 * goodput + w2 * link_trust) - 1
        else:
            reward = -1  # Invalid action
        
        # Move to next flow
        self.current_flow_idx = (self.current_flow_idx + 1) % len(self.flows)
        next_state = self._get_state()
        done = False  # Continuous learning
        
        info = {
            'probing_stats': self.probing_stats.copy()
        }
        
        return next_state, reward, done, info


def train_dqn(train_flows, paths, config):
    """Train DQN agent following Algorithm 1 from simple_dqn.tex"""
    
    # Initialize environment
    env = SelectiveProbingEnvironment(train_flows, paths)
    
    # Initialize networks
    state_dim = 5  # As specified in simple_dqn.tex
    action_dim = len(paths)
    
    online_network = DQN(state_dim, action_dim, config['hidden_sizes'])
    target_network = DQN(state_dim, action_dim, config['hidden_sizes'])
    target_network.load_state_dict(online_network.state_dict())
    
    # Initialize optimizer
    optimizer = optim.Adam(online_network.parameters(), lr=config['learning_rate'])
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(config['buffer_capacity'])
    
    # Training parameters
    epsilon = config['epsilon_start']
    episode_rewards = []
    losses = []
    
    print(f"\nTraining DQN agent...")
    print(f"  State dimensions: {state_dim}")
    print(f"  Action space: {action_dim} paths")
    print(f"  Episodes: {config['n_episodes']}")
    
    # Training loop
    for episode in range(config['n_episodes']):
        if episode % 50 == 0:
            print(f"  Episode {episode}/{config['n_episodes']}")
        state = env.reset()
        episode_reward = 0
        
        # Episode loop
        for step in range(config['episode_length']):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = online_network(torch.FloatTensor(state))
                    action = q_values.argmax().item()
            
            # Take action
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            
            # Train if enough samples
            if len(replay_buffer) >= config['min_replay_size']:
                # Sample minibatch
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = \
                    replay_buffer.sample(config['batch_size'])
                
                # Convert to tensors
                batch_state = torch.FloatTensor(batch_state)
                batch_action = torch.LongTensor(batch_action)
                batch_reward = torch.FloatTensor(batch_reward)
                batch_next_state = torch.FloatTensor(batch_next_state)
                batch_done = torch.FloatTensor(batch_done)
                
                # Compute current Q values
                current_q_values = online_network(batch_state).gather(1, batch_action.unsqueeze(1))
                
                # Compute target Q values
                with torch.no_grad():
                    next_q_values = target_network(batch_next_state).max(1)[0]
                    target_q_values = batch_reward + config['gamma'] * next_q_values * (1 - batch_done)
                
                # Compute loss
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            # Update target network
            if step % config['target_update_freq'] == 0:
                target_network.load_state_dict(online_network.state_dict())
            
            state = next_state
            
            if done:
                break
        
        # Update epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        episode_rewards.append(episode_reward)
    
    # Final statistics
    final_stats = {
        'episode_rewards': episode_rewards,
        'losses': losses,
        'final_epsilon': epsilon,
        'probing_stats': env.probing_stats,
        'avg_reward': np.mean(episode_rewards[-100:]),
        'avg_probes_per_episode': env.probing_stats['bandwidth_probes'] / config['n_episodes']
    }
    
    return online_network, final_stats


def main():
    """Train DQN following simple_dqn.tex approach"""
    
    output_dir = "evaluation_output"
    
    # Load data
    print("Loading data...")
    with open(os.path.join(output_dir, "dense_config.json"), 'r') as f:
        config = json.load(f)
    
    with open(os.path.join(output_dir, "dense_paths.pkl"), 'rb') as f:
        paths = pickle.load(f)
    
    with open(os.path.join(output_dir, "flows_with_performance.pkl"), 'rb') as f:
        all_flows = pickle.load(f)
    
    # Get training flows
    train_flows = [f for f in all_flows if f['start_time'] < 14 * 24 * 60]
    print(f"Training on {len(train_flows)} flows")
    
    # DQN configuration from simple_dqn.tex
    dqn_config = {
        'hidden_sizes': [128, 64],
        'learning_rate': 1e-3,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_capacity': 10000,
        'batch_size': 32,
        'target_update_freq': 100,  # N in Algorithm 1
        'min_replay_size': 1000,
        'n_episodes': 400,  # Increased for better learning
        'episode_length': 100
    }
    
    # Train DQN
    dqn_model, training_stats = train_dqn(train_flows, paths, dqn_config)
    
    # Save model
    print("\nSaving model and results...")
    torch.save({
        'model_state_dict': dqn_model.state_dict(),
        'config': dqn_config,
        'state_dim': 5,
        'action_dim': len(paths)
    }, os.path.join(output_dir, "dqn_model.pth"))
    
    # Save training statistics
    with open(os.path.join(output_dir, "dqn_training_stats.pkl"), 'wb') as f:
        pickle.dump(training_stats, f)
    
    # Update config
    config['dqn_training'] = {
        'n_episodes': dqn_config['n_episodes'],
        'avg_reward': training_stats['avg_reward'],
        'avg_probes_per_episode': training_stats['avg_probes_per_episode'],
        'total_bandwidth_probes': training_stats['probing_stats']['bandwidth_probes']
    }
    
    with open(os.path.join(output_dir, "dense_config.json"), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Average reward (last 100 episodes): {training_stats['avg_reward']:.3f}")
    print(f"  Total bandwidth probes: {training_stats['probing_stats']['bandwidth_probes']}")
    print(f"  Average probes per episode: {training_stats['avg_probes_per_episode']:.1f}")
    print(f"\nModel saved to: {output_dir}/dqn_model.pth")
    print(f"\nNext step: Evaluate all methods with:")
    print(f"  python 04_evaluate_methods.py")


if __name__ == "__main__":
    main()