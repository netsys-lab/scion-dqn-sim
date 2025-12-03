"""
Enhanced DQN agent with Double DQN, action masking, and advanced techniques
Based on state-of-the-art research in RL-based path selection
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Enhanced experience tuple with priority
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 
                        'action_mask', 'next_action_mask'])


@dataclass
class EnhancedDQNConfig:
    """Enhanced DQN hyperparameters"""
    # Network architecture
    hidden_dim: int = 512  # Increased
    n_hidden_layers: int = 4  # Increased
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    # Training parameters
    learning_rate: float = 3e-4
    batch_size: int = 128  # Increased
    gamma: float = 0.99
    tau: float = 0.005  # Increased for faster target updates
    gradient_clip: float = 1.0
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05  # Higher final exploration
    epsilon_decay: float = 0.995
    epsilon_decay_steps: int = 10000
    
    # Memory
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    
    # Update frequency
    update_every: int = 4
    target_update_every: int = 100
    
    # Prioritized replay
    use_prioritized_replay: bool = True
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_annealing_steps: int = 50000
    priority_epsilon: float = 1e-6
    
    # Advanced features
    use_double_dqn: bool = True
    use_dueling_dqn: bool = True
    use_noisy_nets: bool = False
    use_categorical_dqn: bool = False
    
    # Action masking
    use_action_masking: bool = True
    invalid_action_penalty: float = -1e6


class AttentionLayer(nn.Module):
    """Self-attention layer for path features"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = 1.0 / np.sqrt(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores
        scores = torch.bmm(q, k.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        
        # Apply attention
        out = torch.bmm(weights, v)
        return out + x  # Residual connection


class EnhancedDQNNetwork(nn.Module):
    """Enhanced DQN network with attention and advanced features"""
    
    def __init__(self, state_dim: int, action_dim: int, config: EnhancedDQNConfig):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        
        # Feature extraction layers with batch norm
        layers = []
        input_dim = state_dim
        
        for i in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            layers.append(nn.ReLU())
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            input_dim = config.hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Attention layer for path features (optional)
        self.use_attention = True
        if self.use_attention:
            self.attention = AttentionLayer(config.hidden_dim)
        
        if config.use_dueling_dqn:
            # Dueling architecture
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, action_dim)
            )
            
            # Noisy layers (optional)
            if config.use_noisy_nets:
                self._replace_with_noisy_layers()
        else:
            # Standard Q-network
            self.q_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, action_dim)
            )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional action masking
        
        Args:
            state: State tensor
            action_mask: Boolean mask for valid actions
            
        Returns:
            Q-values with masking applied
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Apply attention if enabled
        if self.use_attention and len(features.shape) == 3:
            features = self.attention(features)
            features = features.mean(dim=1)  # Average over sequence
        
        if self.config.use_dueling_dqn:
            # Dueling DQN
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        else:
            # Standard DQN
            q_values = self.q_head(features)
        
        # Apply action masking if provided
        if action_mask is not None and self.config.use_action_masking:
            # Set Q-values for invalid actions to large negative value
            q_values = q_values.masked_fill(~action_mask, self.config.invalid_action_penalty)
        
        return q_values


class PrioritizedReplayBuffer:
    """Enhanced prioritized experience replay buffer"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(self, *args):
        """Add experience with max priority"""
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = Experience(*args)
        self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with importance sampling weights"""
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] + self.epsilon
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights)
        
        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        priorities = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = priorities
    
    def __len__(self):
        return self.size


class EnhancedDQNAgent:
    """Enhanced DQN agent with advanced features"""
    
    def __init__(self, state_dim: int, action_dim: int, config: Optional[EnhancedDQNConfig] = None):
        """Initialize enhanced DQN agent"""
        self.config = config or EnhancedDQNConfig()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.q_network = EnhancedDQNNetwork(state_dim, action_dim, self.config).to(self.device)
        self.target_network = EnhancedDQNNetwork(state_dim, action_dim, self.config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Set to evaluation mode
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=10000,
            gamma=0.9
        )
        
        # Memory
        if self.config.use_prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                self.config.buffer_size,
                self.config.alpha,
                self.config.priority_epsilon
            )
        else:
            self.memory = deque(maxlen=self.config.buffer_size)
        
        # Exploration
        self.epsilon = self.config.epsilon_start
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.beta = self.config.beta_start
        
        # Statistics
        self.losses = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.td_errors = deque(maxlen=1000)
        
        logger.info(f"Initialized enhanced DQN agent with config: {self.config}")
    
    def act(self, state: np.ndarray, 
            valid_actions: Optional[List[int]] = None,
            action_mask: Optional[np.ndarray] = None) -> int:
        """
        Select action using epsilon-greedy policy with action masking
        
        Args:
            state: Current state
            valid_actions: List of valid action indices
            action_mask: Boolean mask for valid actions
            
        Returns:
            Selected action index
        """
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            if valid_actions:
                return random.choice(valid_actions)
            elif action_mask is not None:
                valid_indices = np.where(action_mask)[0]
                if len(valid_indices) > 0:
                    return np.random.choice(valid_indices)
            return random.randint(0, self.action_dim - 1)
        
        # Exploitation with action masking
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if action_mask is not None:
                mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor, mask_tensor)
            else:
                q_values = self.q_network(state_tensor)
                if valid_actions:
                    # Manual masking
                    q_values_np = q_values.cpu().numpy()[0]
                    masked_q = np.full(self.action_dim, -np.inf)
                    for action in valid_actions:
                        masked_q[action] = q_values_np[action]
                    return np.argmax(masked_q)
            
            return q_values.argmax(dim=1).cpu().item()
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                next_state: np.ndarray, done: bool,
                action_mask: Optional[np.ndarray] = None,
                next_action_mask: Optional[np.ndarray] = None):
        """Store experience in replay buffer with action masks"""
        # Default masks if not provided
        if action_mask is None:
            action_mask = np.ones(self.action_dim, dtype=bool)
        if next_action_mask is None:
            next_action_mask = np.ones(self.action_dim, dtype=bool)
        
        if self.config.use_prioritized_replay:
            self.memory.push(state, action, reward, next_state, done,
                           action_mask, next_action_mask)
        else:
            self.memory.append(Experience(state, action, reward, next_state, done,
                                        action_mask, next_action_mask))
    
    def replay(self) -> Optional[float]:
        """Enhanced training step with Double DQN and action masking"""
        if len(self.memory) < self.config.min_buffer_size:
            return None
        
        # Sample batch
        if self.config.use_prioritized_replay:
            experiences, indices, weights = self.memory.sample(
                self.config.batch_size, self.beta
            )
            weights = weights.to(self.device)
        else:
            experiences = random.sample(self.memory, self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)
        action_masks = torch.BoolTensor(np.array([e.action_mask for e in experiences])).to(self.device)
        next_action_masks = torch.BoolTensor(np.array([e.next_action_mask for e in experiences])).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states, action_masks).gather(1, actions.unsqueeze(1))
        
        # Next Q values with Double DQN
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Use online network to select actions
                next_q_values_online = self.q_network(next_states, next_action_masks)
                next_actions = next_q_values_online.argmax(dim=1)
                
                # Use target network to evaluate actions
                next_q_values_target = self.target_network(next_states, next_action_masks)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                # Standard DQN
                next_q_values = self.target_network(next_states, next_action_masks).max(1)[0]
            
            # Compute targets
            target_q_values = rewards + (self.config.gamma * next_q_values * (1 - dones))
        
        # Compute loss with importance sampling weights
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values)
        loss = (weights * F.smooth_l1_loss(
            current_q_values.squeeze(),
            target_q_values,
            reduction='none'
        )).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.gradient_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update priorities if using prioritized replay
        if self.config.use_prioritized_replay and indices is not None:
            td_errors_np = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, td_errors_np)
        
        # Soft update target network
        if self.steps % self.config.target_update_every == 0:
            self._soft_update_target_network()
        
        # Update statistics
        self.steps += 1
        self.losses.append(loss.item())
        self.td_errors.extend(td_errors.detach().cpu().numpy())
        
        # Update beta for prioritized replay
        if self.config.use_prioritized_replay:
            self.beta = min(
                self.config.beta_end,
                self.beta + (self.config.beta_end - self.config.beta_start) / 
                self.config.beta_annealing_steps
            )
        
        return loss.item()
    
    def _soft_update_target_network(self):
        """Soft update of target network parameters (Polyak averaging)"""
        for target_param, local_param in zip(
            self.target_network.parameters(),
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.config.tau * local_param.data +
                (1.0 - self.config.tau) * target_param.data
            )
    
    def update_epsilon(self):
        """Update exploration rate with linear decay"""
        if self.episodes < self.config.epsilon_decay_steps:
            decay_rate = (self.config.epsilon_start - self.config.epsilon_end) / self.config.epsilon_decay_steps
            self.epsilon = self.config.epsilon_start - decay_rate * self.episodes
        else:
            self.epsilon = self.config.epsilon_end
        self.episodes += 1
    
    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episodes': self.episodes,
            'config': self.config
        }, path)
        logger.info(f"Saved enhanced DQN agent to {path}")
    
    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        logger.info(f"Loaded enhanced DQN agent from {path}")
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'avg_td_error': np.mean(self.td_errors) if self.td_errors else 0,
            'beta': self.beta if self.config.use_prioritized_replay else 1.0,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }