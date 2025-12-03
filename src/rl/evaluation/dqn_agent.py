"""
DQN Agent for SCION Path Selection
Designed for evaluation pipeline compatibility
"""

import numpy as np
import pickle
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using NumPy-based DQN")

from .replay_buffer import PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """Configuration for DQN Agent"""
    state_size: int
    action_size: int
    learning_rate: float = 0.001
    gamma: float = 0.95
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    use_dueling: bool = True
    use_double_dqn: bool = True
    use_prioritized_replay: bool = True
    alpha: float = 0.6  # PER alpha
    beta: float = 0.4   # PER beta
    beta_increment: float = 0.001


class DuelingNetwork(nn.Module):
    """Dueling DQN network architecture"""
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = None):
        super(DuelingNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        
        # Shared layers
        self.shared = nn.Sequential()
        prev_size = state_size
        for i, hidden_size in enumerate(hidden_sizes[:-1]):
            self.shared.add_module(f'fc{i}', nn.Linear(prev_size, hidden_size))
            self.shared.add_module(f'relu{i}', nn.ReLU())
            prev_size = hidden_size
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, hidden_sizes[-1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[-1], action_size)
        )
    
    def forward(self, x):
        shared = self.shared(x)
        value = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class NumpyDQN:
    """Numpy-based DQN for when PyTorch is not available"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.weights = {
            'fc1': np.random.randn(state_size, 128) * 0.1,
            'b1': np.zeros(128),
            'fc2': np.random.randn(128, 64) * 0.1,
            'b2': np.zeros(64),
            'fc3': np.random.randn(64, action_size) * 0.1,
            'b3': np.zeros(action_size)
        }
    
    def predict(self, state):
        """Forward pass"""
        if state.ndim == 1:
            state = state.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(state, self.weights['fc1']) + self.weights['b1']
        a1 = np.maximum(0, z1)  # ReLU
        
        # Layer 2
        z2 = np.dot(a1, self.weights['fc2']) + self.weights['b2']
        a2 = np.maximum(0, z2)  # ReLU
        
        # Output layer
        q_values = np.dot(a2, self.weights['fc3']) + self.weights['b3']
        
        return q_values
    
    def update(self, state, action, target):
        """Simple gradient update"""
        q_values = self.predict(state)
        error = target - q_values[0, action]
        
        # Simplified backprop (gradient descent)
        gradient = error * self.learning_rate
        
        # Update weights (simplified)
        self.weights['fc3'][:, action] += gradient * 0.01
        self.weights['b3'][action] += gradient * 0.001


class DQNAgent:
    """DQN Agent for SCION Path Selection"""
    
    def __init__(self, config: DQNConfig):
        self.config = config
        self.epsilon = config.epsilon
        self.step_count = 0
        
        # Initialize memory
        if config.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                config.memory_size,
                config.alpha,
                config.beta,
                config.beta_increment
            )
        else:
            self.memory = []
        
        # Initialize networks
        if TORCH_AVAILABLE and not config.use_dueling:
            # Standard DQN
            self.q_network = nn.Sequential(
                nn.Linear(config.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config.action_size)
            )
            self.target_network = nn.Sequential(
                nn.Linear(config.state_size, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config.action_size)
            )
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
            self.loss_fn = nn.MSELoss()
        elif TORCH_AVAILABLE and config.use_dueling:
            # Dueling DQN
            self.q_network = DuelingNetwork(config.state_size, config.action_size)
            self.target_network = DuelingNetwork(config.state_size, config.action_size)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
            self.loss_fn = nn.MSELoss()
        else:
            # Numpy fallback
            self.q_network = NumpyDQN(config.state_size, config.action_size, config.learning_rate)
            self.target_network = NumpyDQN(config.state_size, config.action_size, config.learning_rate)
        
        # Update target network
        self.update_target_network()
    
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.config.action_size)
        
        if TORCH_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
        else:
            q_values = self.q_network.predict(state)
            return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        if self.config.use_prioritized_replay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) > self.config.memory_size:
                self.memory.pop(0)
    
    def replay(self) -> Optional[float]:
        """Train the model on a batch of experiences"""
        if self.config.use_prioritized_replay:
            if len(self.memory) < self.config.batch_size:
                return None
            
            batch, weights, indices = self.memory.sample(self.config.batch_size)
        else:
            if len(self.memory) < self.config.batch_size:
                return None
            
            indices = np.random.choice(len(self.memory), self.config.batch_size, replace=False)
            batch = [self.memory[i] for i in indices]
            weights = np.ones(self.config.batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        if TORCH_AVAILABLE:
            # Convert to tensors
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            rewards = torch.FloatTensor(rewards)
            weights = torch.FloatTensor(weights)
            
            # Current Q values
            current_q_values = self.q_network(states)
            current_q_values = current_q_values.gather(1, torch.LongTensor(actions).unsqueeze(1))
            
            # Next Q values
            with torch.no_grad():
                if self.config.use_double_dqn:
                    # Double DQN: use online network to select action, target network to evaluate
                    next_actions = self.q_network(next_states).argmax(1)
                    next_q_values = self.target_network(next_states)
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze()
                else:
                    next_q_values = self.target_network(next_states).max(1)[0]
                
                target_q_values = rewards + self.config.gamma * next_q_values * (1 - torch.FloatTensor(dones))
            
            # Compute loss
            loss = (weights * self.loss_fn(current_q_values.squeeze(), target_q_values)).mean()
            
            # Update priorities if using PER
            if self.config.use_prioritized_replay:
                td_errors = (current_q_values.squeeze() - target_q_values).abs().detach().numpy()
                self.memory.update_priorities(indices, td_errors)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_value = loss.item()
        else:
            # Numpy version
            loss_value = 0
            for i in range(self.config.batch_size):
                state = states[i]
                action = actions[i]
                reward = rewards[i]
                next_state = next_states[i]
                done = dones[i]
                
                # Calculate target
                if done:
                    target = reward
                else:
                    next_q = self.target_network.predict(next_state)
                    target = reward + self.config.gamma * np.max(next_q)
                
                # Update
                self.q_network.update(state, action, target)
                
                # Simple loss calculation
                current_q = self.q_network.predict(state)[0, action]
                loss_value += (target - current_q) ** 2
            
            loss_value /= self.config.batch_size
        
        # Decay epsilon
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
        self.step_count += 1
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        if TORCH_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            # Copy numpy weights
            for key in self.q_network.weights:
                self.target_network.weights[key] = self.q_network.weights[key].copy()
    
    def save(self, filepath: str):
        """Save agent to file"""
        agent_data = {
            'config': self.config,
            'epsilon': self.epsilon,
            'step_count': self.step_count
        }
        
        if TORCH_AVAILABLE:
            agent_data['q_network_state'] = self.q_network.state_dict()
            agent_data['target_network_state'] = self.target_network.state_dict()
            agent_data['optimizer_state'] = self.optimizer.state_dict()
        else:
            agent_data['q_network_weights'] = self.q_network.weights
            agent_data['target_network_weights'] = self.target_network.weights
        
        with open(filepath, 'wb') as f:
            pickle.dump(agent_data, f)
    
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            agent_data = pickle.load(f)
        
        self.config = agent_data['config']
        self.epsilon = agent_data['epsilon']
        self.step_count = agent_data['step_count']
        
        if TORCH_AVAILABLE and 'q_network_state' in agent_data:
            self.q_network.load_state_dict(agent_data['q_network_state'])
            self.target_network.load_state_dict(agent_data['target_network_state'])
            self.optimizer.load_state_dict(agent_data['optimizer_state'])
        elif 'q_network_weights' in agent_data:
            self.q_network.weights = agent_data['q_network_weights']
            self.target_network.weights = agent_data['target_network_weights']