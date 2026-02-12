"""
HYBRID DQN Training for TMNF
Learns from BOTH elite experiences AND recent experiences
Prevents learning stagnation while maintaining quality focus
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
import json
from tmnf_env import TMNFEnvironment


class DQNNetwork(nn.Module):
    """Deep Q-Network with better initialization"""
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 64)
        self.fc4 = nn.Linear(64, action_size)
        
        # Better initialization
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class HybridReplayBuffer:
    """
    HYBRID Replay Buffer - FIXES THE LEARNING PARADOX
    
    Maintains TWO buffers:
    1. Elite buffer: Top performing episodes
    2. Recent buffer: Last N episodes regardless of performance
    
    Samples from BOTH to ensure:
    - Learning from good examples (elite)
    - Learning from current policy (recent)
    """
    def __init__(self, capacity=50000, recent_capacity=10000, elite_percentile=20):
        """
        Args:
            capacity: Total capacity for elite transitions
            recent_capacity: Capacity for recent transitions
            elite_percentile: Top X% for elite buffer (increased from 10%)
        """
        self.capacity = capacity
        self.recent_capacity = recent_capacity
        self.elite_percentile = elite_percentile
        
        # ELITE BUFFER - best episodes
        self.elite_episodes = []
        self.elite_transitions = []
        
        # RECENT BUFFER - last N transitions regardless of quality
        self.recent_transitions = deque(maxlen=recent_capacity)
        
        # Tracking
        self.total_episodes_seen = 0
        self.best_score_ever = 0
        self.best_checkpoint_ever = 0
        
    def add_episode(self, transitions, max_checkpoint, episode_length, avg_speed=0.0):
        """Add episode to both elite (if qualified) and recent buffers"""
        
        # Calculate performance score
        checkpoint_score = max_checkpoint * 1000
        speed_bonus = avg_speed * 2
        time_bonus = max(0, 3000 - episode_length)
        performance_score = checkpoint_score + speed_bonus + time_bonus
        
        self.total_episodes_seen += 1
        
        # Track best ever
        if performance_score > self.best_score_ever:
            self.best_score_ever = performance_score
            print(f"\n🏆 NEW BEST: Score={performance_score:.0f} "
                  f"(CP={max_checkpoint}, Steps={episode_length}, Speed={avg_speed:.1f})")
        
        if max_checkpoint > self.best_checkpoint_ever:
            self.best_checkpoint_ever = max_checkpoint
            print(f"  🎯 New best checkpoint: {max_checkpoint}")
        
        # ALWAYS add to recent buffer
        for transition in transitions:
            self.recent_transitions.append(transition)
        
        # Add to elite buffer
        self.elite_episodes.append({
            'score': performance_score,
            'transitions': transitions,
            'checkpoint': max_checkpoint,
            'length': episode_length,
            'avg_speed': avg_speed
        })
        
        # Update elite buffer
        self._update_elite_buffer()
    
    def _update_elite_buffer(self):
        """Keep only top elite_percentile% of episodes in elite buffer"""
        if len(self.elite_episodes) < 10:
            # Keep all episodes until we have enough
            self.elite_transitions = []
            for ep in self.elite_episodes:
                self.elite_transitions.extend(ep['transitions'])
            return
        
        # Sort by performance
        self.elite_episodes.sort(key=lambda x: x['score'], reverse=True)
        
        # Keep top percentile
        num_elite = max(5, int(len(self.elite_episodes) * (self.elite_percentile / 100)))
        self.elite_episodes = self.elite_episodes[:num_elite]
        
        # Flatten elite transitions
        self.elite_transitions = []
        for ep in self.elite_episodes:
            self.elite_transitions.extend(ep['transitions'])
        
        # Limit capacity
        if len(self.elite_transitions) > self.capacity:
            self.elite_transitions = self.elite_transitions[-self.capacity:]
        
        # Stats
        if len(self.elite_episodes) > 0:
            elite_cps = [ep['checkpoint'] for ep in self.elite_episodes]
            elite_speeds = [ep['avg_speed'] for ep in self.elite_episodes]
            print(f"  📊 Elite: {len(self.elite_episodes)} eps, "
                  f"CP: {max(elite_cps)}/{np.mean(elite_cps):.1f}, "
                  f"Speed: {max(elite_speeds):.1f}/{np.mean(elite_speeds):.1f}")
    
    def sample(self, batch_size, elite_ratio=0.5):
        """
        Sample from BOTH elite and recent buffers
        
        Args:
            batch_size: Total batch size
            elite_ratio: Fraction of batch from elite buffer (default 50/50)
        """
        # Calculate samples from each buffer
        num_elite = int(batch_size * elite_ratio)
        num_recent = batch_size - num_elite
        
        # Sample from elite buffer
        elite_batch = []
        if len(self.elite_transitions) >= num_elite and num_elite > 0:
            elite_batch = random.sample(self.elite_transitions, num_elite)
        elif len(self.elite_transitions) > 0:
            # If not enough, take what we can
            elite_batch = random.sample(self.elite_transitions, 
                                       min(len(self.elite_transitions), num_elite))
        
        # Sample from recent buffer
        recent_batch = []
        if len(self.recent_transitions) >= num_recent and num_recent > 0:
            recent_batch = random.sample(list(self.recent_transitions), num_recent)
        elif len(self.recent_transitions) > 0:
            # If not enough, take what we can
            recent_batch = random.sample(list(self.recent_transitions),
                                        min(len(self.recent_transitions), num_recent))
        
        # Combine batches
        combined_batch = elite_batch + recent_batch
        
        if len(combined_batch) < batch_size // 2:
            # Not enough data yet
            return None
        
        states, actions, rewards, next_states, dones = zip(*combined_batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def get_stats(self):
        """Get buffer statistics"""
        if not self.elite_episodes:
            return {
                'elite_episodes': 0,
                'elite_transitions': 0,
                'recent_transitions': len(self.recent_transitions),
                'total_episodes': self.total_episodes_seen
            }
        
        elite_cps = [ep['checkpoint'] for ep in self.elite_episodes]
        elite_speeds = [ep['avg_speed'] for ep in self.elite_episodes]
        
        return {
            'elite_episodes': len(self.elite_episodes),
            'elite_transitions': len(self.elite_transitions),
            'recent_transitions': len(self.recent_transitions),
            'total_episodes': self.total_episodes_seen,
            'elite_best_cp': max(elite_cps),
            'elite_avg_cp': float(np.mean(elite_cps)),
            'elite_best_speed': max(elite_speeds),
            'elite_avg_speed': float(np.mean(elite_speeds)),
            'best_score_ever': self.best_score_ever
        }
    
    def __len__(self):
        return len(self.elite_transitions) + len(self.recent_transitions)


class DQNAgent:
    """DQN Agent with hybrid learning"""
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.10  # Increased from 0.05 - maintain exploration
        self.epsilon_decay = 0.9995  # Slower decay
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.update_target_every = 1000  # Less frequent updates
        self.update_counter = 0
    
    def select_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def train(self, replay_buffer, batch_size, elite_ratio=0.5):
        """
        Train on hybrid batch
        
        Args:
            elite_ratio: Fraction from elite buffer (0.5 = 50/50 split)
        """
        batch = replay_buffer.sample(batch_size, elite_ratio=elite_ratio)
        if batch is None:
            return 0.0
        
        states, actions, rewards, next_states, dones = batch
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = self.loss_fn(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        self.update_counter += 1
        if self.update_counter % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath):
        """Save agent"""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter
        }
        torch.save(checkpoint, filepath)
        print(f"✓ Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', 0.1)  # Reset to higher epsilon
            self.update_counter = checkpoint.get('update_counter', 0)
            print(f"✓ Agent loaded from {filepath}")
            print(f"  Epsilon reset to: {self.epsilon}")
            return True
        return False


def evaluate_agent(env, agent, num_episodes=5):
    """Evaluate agent without exploration"""
    eval_rewards = []
    eval_checkpoints = []
    eval_lengths = []
    eval_speeds = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        max_checkpoints = 0
        episode_length = 0
        episode_speeds = []
        
        while True:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            max_checkpoints = max(max_checkpoints, info.get('checkpoint', 0))
            episode_speeds.append(info.get('speed', 0.0))
            
            obs = next_obs
            if terminated or truncated:
                break
        
        eval_rewards.append(episode_reward)
        eval_checkpoints.append(max_checkpoints)
        eval_lengths.append(episode_length)
        eval_speeds.append(np.mean(episode_speeds) if episode_speeds else 0.0)
    
    return (np.mean(eval_rewards), np.mean(eval_checkpoints), 
            np.mean(eval_lengths), np.mean(eval_speeds))


def train_dqn(env, agent, num_episodes=3000, batch_size=64, save_every=50):
    """
    HYBRID Training Loop
    Learns from both elite and recent experiences
    """
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Hybrid replay buffer
    replay_buffer = HybridReplayBuffer(
        capacity=50000,           # Elite buffer size
        recent_capacity=15000,    # Recent buffer size (increased)
        elite_percentile=20       # Keep top 20% (more permissive)
    )
    
    episode_rewards = []
    episode_lengths = []
    episode_checkpoints = []
    episode_speeds = []
    eval_rewards = []
    losses = []
    
    # Tracking success rate
    recent_completions = deque(maxlen=50)  # Track last 50 episodes
    
    print("\n" + "="*70)
    print("Starting HYBRID DQN Training")
    print("Strategy: Learn from BOTH elite (best runs) AND recent (current policy)")
    print("This prevents learning stagnation!")
    print("="*70)
    
    for episode in range(1, num_episodes + 1):
        # Dynamic elite ratio - start balanced, increase elite over time
        if episode < 500:
            elite_ratio = 0.3  # 30% elite, 70% recent - focus on current policy
        elif episode < 1500:
            elite_ratio = 0.5  # 50/50 split
        else:
            elite_ratio = 0.6  # 60% elite, 40% recent - focus on good examples
        
        # Adjust max steps
        if episode < 300:
            env.MAX_EPISODE_STEPS = 5000
        elif episode < 800:
            env.MAX_EPISODE_STEPS = 4000
        else:
            env.MAX_EPISODE_STEPS = 3000
        
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        max_checkpoints = 0
        
        episode_transitions = []
        episode_speed_list = []
        
        while True:
            action = agent.select_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            max_checkpoints = max(max_checkpoints, info.get('checkpoint', 0))
            
            current_speed = info.get('speed', 0.0)
            episode_speed_list.append(current_speed)
            
            episode_transitions.append((obs, action, reward, next_obs, terminated))
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        avg_speed = np.mean(episode_speed_list) if episode_speed_list else 0.0
        
        # Add to buffer
        replay_buffer.add_episode(
            transitions=episode_transitions,
            max_checkpoint=max_checkpoints,
            episode_length=episode_length,
            avg_speed=avg_speed
        )
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_checkpoints.append(max_checkpoints)
        episode_speeds.append(avg_speed)
        
        # Track completion rate
        completed = (max_checkpoints >= 9)
        recent_completions.append(completed)
        completion_rate = sum(recent_completions) / len(recent_completions) if recent_completions else 0.0
        
        # TRAIN with hybrid sampling
        if len(replay_buffer) > 1000:
            num_training_steps = 5  # Reduced from 10
            for _ in range(num_training_steps):
                loss = agent.train(replay_buffer, batch_size, elite_ratio=elite_ratio)
                if loss > 0:
                    losses.append(loss)
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            avg_checkpoints = np.mean(episode_checkpoints[-10:])
            avg_episode_speed = np.mean(episode_speeds[-10:])
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0.0
            
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:8.2f} | Checkpoints: {avg_checkpoints:4.2f} | Speed: {avg_episode_speed:5.1f} km/h")
            print(f"  Completion Rate (last 50): {completion_rate*100:5.1f}% | Epsilon: {agent.epsilon:.4f}")
            print(f"  Elite Ratio: {elite_ratio*100:.0f}% | Loss: {avg_loss:7.4f}")
            
            # Buffer stats
            stats = replay_buffer.get_stats()
            print(f"  Buffers: Elite={stats['elite_transitions']}, Recent={stats['recent_transitions']}")
            if stats.get('elite_best_cp', 0) > 0:
                print(f"  Elite Best: CP={stats['elite_best_cp']}, Speed={stats['elite_best_speed']:.1f} km/h")
        
        # Evaluation
        if episode % 50 == 0:
            eval_reward, eval_checkpoint, eval_length, eval_speed = evaluate_agent(env, agent, num_episodes=5)
            eval_rewards.append(eval_reward)
            eval_completions = sum(1 for _ in range(5) if eval_checkpoint >= 9) / 5
            print(f"  ═══ EVAL ═══")
            print(f"  Reward: {eval_reward:7.2f} | CP: {eval_checkpoint:4.2f} | Speed: {eval_speed:5.1f} km/h")
            print(f"  Completion: {eval_completions*100:.0f}%")
        
        # Save checkpoint
        if episode % save_every == 0:
            agent.save(f"checkpoints/agent_ep{episode}.pt")
            
            stats = replay_buffer.get_stats()
            log_stats = {
                'episode': int(episode),
                'episode_rewards': [float(x) for x in episode_rewards[-100:]],
                'episode_checkpoints': [float(x) for x in episode_checkpoints[-100:]],
                'episode_speeds': [float(x) for x in episode_speeds[-100:]],
                'completion_rate': float(completion_rate),
                'buffer_stats': stats,
                'epsilon': float(agent.epsilon),
                'elite_ratio': float(elite_ratio)
            }
            with open(f"logs/log_ep{episode}.json", 'w') as f:
                json.dump(log_stats, f, indent=2)
    
    # Final save
    agent.save("checkpoints/agent_final.pt")
    print("\n✓ Training completed!")
    print(f"✓ Final completion rate: {completion_rate*100:.1f}%")


if __name__ == "__main__":
    env = TMNFEnvironment()
    
    if not env.connect_to_game():
        print("✗ Cannot connect to game!")
        exit(1)
    
    agent = DQNAgent(
        state_size=6,
        action_size=5,
        lr=5e-5,         # Slightly lower LR
        gamma=0.99,
        epsilon=1.0
    )
    
    # Load with epsilon reset
    if agent.load("checkpoints/agent_final.pt"):
        print("✓ Resuming training (epsilon reset for exploration)")
        agent.epsilon = 0.3  # Force more exploration
    else:
        print("✓ Starting fresh training")
    
    try:
        train_dqn(
            env, 
            agent, 
            num_episodes=3000, 
            batch_size=64, 
            save_every=50
        )
    except KeyboardInterrupt:
        print("\n✓ Training stopped by user")
        agent.save("checkpoints/agent_final.pt")
    finally:
        env.close()