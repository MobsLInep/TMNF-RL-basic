"""
Utility functions for TMNF training: evaluation, visualization, analysis
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tmnf_env import TMNFEnvironment
from train_dqn import DQNAgent


class ModelEvaluator:
    """Evaluate trained models without learning"""
    
    def __init__(self, agent, env, num_episodes=10):
        self.agent = agent
        self.env = env
        self.num_episodes = num_episodes
        self.device = agent.device
    
    def evaluate(self):
        """Run evaluation episodes"""
        results = {
            'episodes': [],
            'rewards': [],
            'lengths': [],
            'checkpoints': [],
            'success_rate': 0.0
        }
        
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # Greedy only
        
        print("\n" + "="*60)
        print("Model Evaluation")
        print("="*60)
        
        finished = 0
        
        for ep in range(1, self.num_episodes + 1):
            obs, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            max_checkpoint = 0
            finished_race = False
            
            while True:
                action = self.agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                max_checkpoint = max(max_checkpoint, info.get('checkpoint', 0))
                
                if info.get('termination_reason') == 'finish_line':
                    finished_race = True
                
                obs = next_obs
                
                if terminated or truncated:
                    break
            
            results['episodes'].append(ep)
            results['rewards'].append(episode_reward)
            results['lengths'].append(episode_length)
            results['checkpoints'].append(max_checkpoint)
            
            if finished_race:
                finished += 1
            
            status = "✓ FINISHED" if finished_race else "✗ Not finished"
            print(f"Episode {ep:2d}: reward={episode_reward:7.2f}, "
                  f"length={episode_length:4d}, checkpoints={max_checkpoint:2d} {status}")
        
        results['success_rate'] = finished / self.num_episodes
        results['finish_count'] = finished
        
        print(f"\n✓ Success rate: {results['success_rate']*100:.1f}% ({finished}/{self.num_episodes})")
        print(f"✓ Avg reward: {np.mean(results['rewards']):.2f}")
        print(f"✓ Avg length: {np.mean(results['lengths']):.1f}")
        print(f"✓ Avg checkpoints: {np.mean(results['checkpoints']):.2f}")
        
        self.agent.epsilon = original_epsilon
        return results


class TrainingVisualizer:
    """Visualize training progress"""
    
    @staticmethod
    def plot_training_curves(log_dir="logs"):
        """Plot training curves from logs"""
        
        if not os.path.exists(log_dir):
            print(f"✗ Log directory not found: {log_dir}")
            return
        
        # Collect data from all logs
        episodes = []
        rewards = []
        lengths = []
        checkpoints = []
        
        for log_file in sorted(Path(log_dir).glob("log_*.json")):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                episodes.append(data['episode'])
                rewards.append(data['avg_reward'])
                lengths.append(data['avg_length'])
                checkpoints.append(data['avg_checkpoints'])
            except:
                pass
        
        if not episodes:
            print("✗ No log files found")
            return
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('TMNF DQN Training Progress', fontsize=16, fontweight='bold')
        
        # Plot 1: Rewards
        axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Reward')
        axes[0, 0].set_title('Reward Progression')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Episode Length
        axes[0, 1].plot(episodes, lengths, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Length (steps)')
        axes[0, 1].set_title('Episode Length Progression')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Checkpoints Reached
        axes[1, 0].plot(episodes, checkpoints, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Average Checkpoints')
        axes[1, 0].set_title('Checkpoint Progress')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Statistics
        axes[1, 1].axis('off')
        stats_text = f"""
Training Statistics (Last 10 Episodes)

Episodes Trained: {episodes[-1] if episodes else 0}
Final Avg Reward: {rewards[-1]:.2f}
Final Avg Length: {lengths[-1]:.1f}
Final Avg Checkpoints: {checkpoints[-1]:.2f}

Best Reward: {max(rewards):.2f}
Best Checkpoints: {max(checkpoints):.2f}

Improvement:
  Reward: {rewards[-1] - rewards[0]:.2f}
  Checkpoints: {checkpoints[-1] - checkpoints[0]:.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
        print("✓ Training curves saved to training_curves.png")
        plt.show()


class ConfigManager:
    """Manage training configurations"""
    
    @staticmethod
    def save_config(config, filepath="config.json"):
        """Save configuration"""
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Config saved to {filepath}")
    
    @staticmethod
    def load_config(filepath="config.json"):
        """Load configuration"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None
    
    @staticmethod
    def default_config():
        """Get default configuration"""
        return {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.05,
            'epsilon_decay': 0.995,
            'batch_size': 32,
            'buffer_capacity': 10000,
            'update_target_every': 1000,
            'num_episodes': 2000,
            'save_every': 50,
            'checkpoint_data': [
                [704, 560], [683, 557], [664, 549], [646, 536],
                [634, 520], [627, 503], [624, 483], [620, 461],
                [609, 446], [594, 435.5]
            ]
        }


def test_environment():
    """Test environment connectivity and basic functionality"""
    
    print("\n" + "="*60)
    print("Testing TMNF Environment")
    print("="*60)
    
    env = TMNFEnvironment()
    
    # Test 1: Connection
    print("\n1. Testing connection to game...")
    if not env.connect_to_game():
        print("✗ Connection failed!")
        return False
    print("✓ Connected successfully")
    
    # Test 2: Reset
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Initial obs: {obs}")
    
    # Test 3: Step
    print("\n3. Testing step function...")
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0:
            print(f"  Step {step:2d}: reward={reward:7.3f}, speed={info['speed']:6.1f} km/h, "
                  f"checkpoint={info['checkpoint']}, terminated={terminated}")
        
        if terminated:
            print(f"✓ Episode terminated naturally: {info.get('termination_reason')}")
            break
    
    print("✓ Step function working correctly")
    
    # Test 4: Telemetry
    print("\n4. Testing telemetry parsing...")
    print(f"✓ Position: ({obs[0]:.1f}, {obs[1]:.1f})")
    print(f"✓ Speed: {obs[2]:.1f} km/h")
    print(f"✓ Checkpoint: {int(obs[3])}")
    print(f"✓ Velocity: ({obs[4]:.2f}, {obs[5]:.2f})")
    
    env.close()
    print("\n✓ All tests passed!")
    return True


def main():
    """Interactive menu"""
    
    while True:
        print("\n" + "="*60)
        print("TMNF Training Utilities")
        print("="*60)
        print("1. Test environment connectivity")
        print("2. Evaluate trained model")
        print("3. Plot training curves")
        print("4. Save/Load configuration")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            test_environment()
        
        elif choice == "2":
            model_path = input("Enter model path (default: checkpoints/agent_final.pt): ").strip()
            if not model_path:
                model_path = "checkpoints/agent_final.pt"
            
            if not os.path.exists(model_path):
                print(f"✗ Model not found: {model_path}")
                continue
            
            env = TMNFEnvironment()
            if not env.connect_to_game():
                print("✗ Cannot connect to game")
                continue
            
            agent = DQNAgent(state_size=6, action_size=5)
            agent.load(model_path)
            
            evaluator = ModelEvaluator(agent, env, num_episodes=10)
            evaluator.evaluate()
            
            env.close()
        
        elif choice == "3":
            TrainingVisualizer.plot_training_curves()
        
        elif choice == "4":
            action = input("Save (s) or Load (l) config? ").strip().lower()
            if action == 's':
                config = ConfigManager.default_config()
                ConfigManager.save_config(config)
            elif action == 'l':
                config = ConfigManager.load_config()
                if config:
                    print(json.dumps(config, indent=2))
        
        elif choice == "5":
            print("Exiting...")
            break


if __name__ == "__main__":
    main()
