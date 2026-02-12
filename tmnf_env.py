"""
TMNF Custom Gymnasium Environment - SPEED-FOCUSED VERSION
Prioritizes fast completion with speed bonuses and relaxed brake penalties
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import socket
import json
import time
import struct
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import pickle
import os


@dataclass
class Checkpoint:
    """Checkpoint data with bounding box"""
    index: int
    x: float
    z: float
    tolerance: float = 1.0
    is_finish_line: bool = False
    finish_points: Optional[List[Tuple[float, float]]] = None


class TMNFEnvironment(gym.Env):
    """
    Speed-focused TMNF gymnasium environment
    Heavily rewards fast completion and maintaining speed
    """
    
    metadata = {'render_modes': []}
    
    # SPEED-FOCUSED REWARD WEIGHTS
    REWARDS = {
        'time_penalty': -0.01,               # 10x stronger - time matters!
        'speed_bonus': 0.002,                # NEW: reward for maintaining speed
        'distance_decrease': 0.1,            # Small progress reward
        'distance_increase': -0.05,          # Small wrong-direction penalty
        'checkpoint_reached': 5.0,           # Higher base checkpoint reward
        'checkpoint_speed_bonus': 3.0,       # NEW: bonus for fast checkpoint
        'finish_line': 100.0,                # Much higher finish reward
        'finish_speed_bonus': 50.0,          # NEW: bonus for fast finish
        'collision': -2.0,                   # Heavier crash penalty
        'wall_climb': -5.0,                  # Heavier wall penalty
        'stuck': -2.0,                       # Heavier stuck penalty
        'brake_hold': -0.05,                 # REDUCED - less harsh
        'brake_spam': -0.05,                 # REDUCED - allow tactical braking
    }
    
    # Timing/physics constants
    SKIP_FRAMES = 1
    ACTION_FRAME_SKIP = 1
    STUCK_SPEED_THRESHOLD = 10.0
    STUCK_TIME_THRESHOLD = 4.0
    CRASH_SPEED_LOSS = 0.35
    WALL_CLIMB_HEIGHT = 11.0
    MAX_EPISODE_STEPS = 3000
    MAX_EPISODE_TIME = 240.0
    
    def __init__(self, checkpoint_data: List[Tuple[float, float]] = None):
        """Initialize environment"""
        self.socket = None
        self.checkpoints = self._init_checkpoints(checkpoint_data)
        self.current_checkpoint_idx = 0
        
        # State tracking
        self.last_position = np.array([0.0, 0.0])
        self.last_speed = 0.0
        self.speed_history = deque(maxlen=10)
        self.stuck_timer = 0.0
        self.episode_steps = 0
        self.frame_counter = 0
        self.episode_start_time = time.time()
        
        # Distance tracking
        self.last_distance_to_checkpoint = float('inf')
        self.consecutive_distance_increases = 0
        self.backward_movement_count = 0
        
        # Brake tracking (continuous hold)
        self.brake_hold_start_time = None
        self.brake_hold_threshold = 2.0  # Increased from 0.75s - allow longer braking
        
        # Brake tap spam detection (rapid tapping)
        self.brake_tap_times = deque(maxlen=30)  # Track more taps
        self.brake_tap_threshold = 25  # Increased from 10 - allow more tactical braking
        self.brake_tap_window = 1.0  # Within 1 second
        self.last_action_was_brake = False
        
        # Speed tracking for rewards
        self.checkpoint_times = []
        self.last_checkpoint_time = time.time()
        self.episode_speeds = []
        
        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)
        
        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([-1000.0, -1000.0, 0.0, 0.0, -100.0, -100.0]),
            high=np.array([1000.0, 1000.0, 300.0, len(self.checkpoints), 100.0, 100.0]),
            dtype=np.float32
        )
        
        self.last_obs = None
        self.info_dict = {}

        # Learning / startup control
        self.learning_started = False
        
        # Logging
        self.episode_reward_log = []
        
    def _init_checkpoints(self, checkpoint_data) -> List[Checkpoint]:
        """Initialize checkpoints from data"""
        if checkpoint_data is None:
            # Default TMNF track
            checkpoint_data = [
                (704, 560), (683, 557), (664, 549), (646, 536),
                (634, 520), (627, 503), (624, 483), (620, 461),
                (609, 446), (594, 435.5)
            ]
        
        checkpoints = []
        for idx, (x, z) in enumerate(checkpoint_data[:-1]):
            checkpoints.append(Checkpoint(
                index=idx,
                x=x,
                z=z,
                tolerance=1.0,
                is_finish_line=False
            ))
        
        # Add finish line
        finish_points = [
            (565.753, 442.545),
            (565.648, 421.224)
        ]
        checkpoints.append(Checkpoint(
            index=len(checkpoint_data) - 1,
            x=np.mean([p[0] for p in finish_points]),
            z=np.mean([p[1] for p in finish_points]),
            tolerance=2.0,
            is_finish_line=True,
            finish_points=finish_points
        ))
        
        return checkpoints
    
    def connect_to_game(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 5.0) -> bool:
        """Connect to TMInterface plugin"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(timeout)
            self.socket.connect((host, port))
            print(f"✓ Connected to game at {host}:{port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def _read_telemetry(self) -> Optional[Dict]:
        """Read telemetry frame from socket"""
        if self.socket is None:
            return None
        
        try:
            self.socket.settimeout(0.01)
            data = self.socket.recv(4096).decode('utf-8')
            if data:
                lines = data.strip().split('\n')
                for line in reversed(lines):
                    if line:
                        try:
                            obj = json.loads(line)
                            if 'position' in obj:
                                return obj
                        except json.JSONDecodeError:
                            pass
        except socket.timeout:
            pass
        except Exception as e:
            print(f"Telemetry read error: {e}")
        
        return None
    
    def _send_action(self, action: int):
        """Send action to game"""
        if self.socket is None:
            return
        
        # Action: 0=noop, 1=left, 2=right, 3=accel, 4=brake
        left = action == 1
        right = action == 2
        accel = action == 3
        brake = action == 4
        
        payload = struct.pack("<iBBBB",
                            1,  # MSG_TYPE_SET_INPUT
                            1 if left else 0,
                            1 if right else 0,
                            1 if accel else 0,
                            1 if brake else 0)
        try:
            self.socket.sendall(payload)
        except:
            pass
    
    def _send_reset_command(self, reason: str = "Episode ended"):
        """Send reset command to plugin"""
        if self.socket is None:
            print("✗ Cannot reset: Socket not connected")
            return False
        
        try:
            payload = struct.pack("<i", 2)  # CResetRace
            self.socket.sendall(payload)
            print(f"✓ Reset command sent to game ({reason})")
            time.sleep(0.3)  # Give game time to reset
            return True
        except Exception as e:
            print(f"✗ Reset failed: {e}")
            return False
    
    def _get_observation(self, telemetry: Dict) -> Tuple[np.ndarray, Tuple[float, float], float]:
        """Extract observation from telemetry"""
        pos = telemetry.get('position', {})
        x = float(pos.get('x', 0))
        z = float(pos.get('z', 0))
        
        velocity = telemetry.get('velocity', {})
        vx = float(velocity.get('x', 0))
        vz = float(velocity.get('z', 0))
        speed_kmh = np.sqrt(vx**2 + vz**2) * 3.6
        
        # Current checkpoint target
        if self.current_checkpoint_idx < len(self.checkpoints):
            cp = self.checkpoints[self.current_checkpoint_idx]
            dx = x - cp.x
            dz = z - cp.z
        else:
            dx, dz = 0.0, 0.0
        
        # Observation: [x, z, speed, checkpoint_idx, dx, dz]
        obs = np.array([x, z, speed_kmh, self.current_checkpoint_idx, dx, dz], dtype=np.float32)
        
        return obs, (x, z), speed_kmh
    
    def _calculate_distance_reward(self, distance_current: float, last_distance: float) -> float:
        """Calculate smooth distance-based reward"""
        if last_distance == float('inf'):
            return 0.0
        
        distance_delta = last_distance - distance_current
        
        if distance_delta > 0:
            # Getting closer
            reward = min(distance_delta * 0.05, 0.1)
        else:
            # Getting farther
            reward = max(distance_delta * 0.05, -0.1)
        
        return reward
    
    def _compute_reward(self, telemetry: Dict, obs: np.ndarray, action: int, obs_before: np.ndarray = None) -> Tuple[float, bool]:
        """
        Speed-focused reward computation
        Heavily rewards maintaining speed and fast completion
        """
        reward = 0.0
        terminated = False
        
        x = float(telemetry.get('position', {}).get('x', 0))
        z = float(telemetry.get('position', {}).get('z', 0))
        
        velocity = telemetry.get('velocity', {})
        vx = float(velocity.get('x', 0))
        vz = float(velocity.get('z', 0))
        speed_kmh = np.sqrt(vx**2 + vz**2) * 3.6
        
        # Track speed for episode averaging
        self.episode_speeds.append(speed_kmh)
        
        # STRONGER TIME PENALTY
        reward += self.REWARDS['time_penalty']
        
        # NEW: SPEED BONUS - reward maintaining high speed
        # Scale reward based on speed (max at 200+ km/h)
        speed_normalized = min(speed_kmh / 200.0, 1.0)
        reward += self.REWARDS['speed_bonus'] * speed_normalized
        
        # WALL CLIMB CHECK
        y = float(telemetry.get('position', {}).get('y', 9.0))
        if y > self.WALL_CLIMB_HEIGHT:
            reward += self.REWARDS['wall_climb']
            terminated = True
            self.info_dict['termination_reason'] = 'wall_climb'
            return reward, terminated
        
        # CRASH CHECK (speed drop)
        speed_max = max(self.speed_history) if self.speed_history else speed_kmh
        if speed_max > 50 and speed_kmh < speed_max * (1 - self.CRASH_SPEED_LOSS):
            reward += self.REWARDS['collision']
        
        # STUCK CHECK
        if speed_kmh < self.STUCK_SPEED_THRESHOLD:
            self.stuck_timer += 0.05
            if self.stuck_timer > self.STUCK_TIME_THRESHOLD:
                reward += self.REWARDS['stuck']
                terminated = True
                self.info_dict['termination_reason'] = 'stuck'
                return reward, terminated
        else:
            self.stuck_timer = 0.0
        
        # RELAXED BRAKE HOLD PENALTY - only penalize if held >2 seconds (was 0.75s)
        if action == 4:
            if self.brake_hold_start_time is None:
                self.brake_hold_start_time = time.time()
            else:
                brake_duration = time.time() - self.brake_hold_start_time
                if brake_duration > self.brake_hold_threshold:
                    # Gentle penalty for holding brake
                    reward += self.REWARDS['brake_hold'] * (brake_duration - self.brake_hold_threshold)
        else:
            self.brake_hold_start_time = None
        
        # RELAXED BRAKE TAP SPAM - only penalize excessive spam
        current_time = time.time()
        
        if action == 4 and not self.last_action_was_brake:
            self.brake_tap_times.append(current_time)
            recent_taps = [t for t in self.brake_tap_times if current_time - t <= self.brake_tap_window]
            
            # Only start penalizing at 15+ taps (was 5)
            if len(recent_taps) >= 15:
                tap_penalty = self.REWARDS['brake_spam'] * (len(recent_taps) - 14)
                reward += tap_penalty
            
            # Only terminate at 25+ taps (was 10)
            if len(recent_taps) >= self.brake_tap_threshold:
                terminated = True
                self.info_dict['termination_reason'] = 'excessive_brake_spam'
                print(f"  ⚠️ Excessive brake spam: {len(recent_taps)} taps in {self.brake_tap_window}s")
                self.episode_reward_log.append(-10.0)
                return -10.0, terminated  # Reduced from -50
        
        self.last_action_was_brake = (action == 4)
        
        # DISTANCE-BASED REWARD
        if self.current_checkpoint_idx < len(self.checkpoints):
            current_cp = self.checkpoints[self.current_checkpoint_idx]
            distance_current = np.sqrt((x - current_cp.x)**2 + (z - current_cp.z)**2)
            
            distance_reward = self._calculate_distance_reward(
                distance_current,
                self.last_distance_to_checkpoint
            )
            reward += distance_reward
            self.last_distance_to_checkpoint = distance_current
        
        # FINISH LINE HANDLING - Enhanced with speed bonus
        if self.current_checkpoint_idx == len(self.checkpoints) - 1:
            checkpoint = self.checkpoints[self.current_checkpoint_idx]
            if checkpoint.is_finish_line:
                p1, p2 = checkpoint.finish_points
                t = (z - p1[1]) / (p2[1] - p1[1]) if p2[1] != p1[1] else 0
                t = np.clip(t, 0, 1)
                closest_x = p1[0] + t * (p2[0] - p1[0])
                closest_z = p1[1] + t * (p2[1] - p1[1])
                dist = np.sqrt((x - closest_x)**2 + (z - closest_z)**2)
                
                if dist <= checkpoint.tolerance:
                    # Base finish reward
                    finish_reward = self.REWARDS['finish_line']
                    
                    # BONUS for finishing with high speed
                    if speed_kmh > 100:
                        speed_bonus = self.REWARDS['finish_speed_bonus'] * (speed_kmh / 200.0)
                        finish_reward += speed_bonus
                        print(f"  ★★★ FINISH! Speed={speed_kmh:.1f} km/h, Speed bonus: +{speed_bonus:.1f}")
                    else:
                        print(f"  ★★★ FINISH! Speed={speed_kmh:.1f} km/h")
                    
                    reward += finish_reward
                    terminated = True
                    self.info_dict['termination_reason'] = 'finish_line'
                    self.episode_reward_log.append(finish_reward)
                    return reward, terminated
        
        # EPISODE TIMEOUT
        elapsed_time = time.time() - self.episode_start_time
        if elapsed_time > self.MAX_EPISODE_TIME:
            reward -= 0.5  # Stronger timeout penalty
            terminated = True
            self.info_dict['termination_reason'] = 'timeout'
            return reward, terminated
        
        # EPISODE LENGTH TERMINATION
        self.episode_steps += 1
        if self.episode_steps >= self.MAX_EPISODE_STEPS:
            terminated = True
            self.info_dict['termination_reason'] = 'max_steps'
        
        self.speed_history.append(speed_kmh)
        self.episode_reward_log.append(reward)
        
        return reward, terminated
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        super().reset(seed=seed)
        
        # Determine reset reason
        reset_reason = self.info_dict.get('termination_reason', 'Episode ended')
        
        # Format reason with details
        last_speed = self.last_speed
        stuck_time = self.stuck_timer
        steps_taken = self.episode_steps
        checkpoint_reached = self.current_checkpoint_idx
        avg_speed = np.mean(self.episode_speeds) if self.episode_speeds else 0.0
        
        if reset_reason == 'stuck':
            reset_reason = f"Stuck for {stuck_time:.1f}s at speed {last_speed:.1f} km/h"
        elif reset_reason == 'wall_climb':
            reset_reason = f"Wall climb detected (height > {self.WALL_CLIMB_HEIGHT}m)"
        elif reset_reason == 'timeout':
            elapsed = time.time() - self.episode_start_time
            reset_reason = f"Timeout after {elapsed:.1f}s"
        elif reset_reason == 'max_steps':
            reset_reason = f"Max steps reached ({steps_taken} steps)"
        elif reset_reason == 'finish_line':
            reset_reason = f"🏁 FINISH LINE! Checkpoint {checkpoint_reached}, Avg Speed {avg_speed:.1f} km/h"
        elif reset_reason == 'excessive_brake_spam':
            reset_reason = f"Excessive brake spam ({self.brake_tap_threshold}+ taps)"
        else:
            reset_reason = "New episode started"
        
        # Log episode summary
        if self.episode_reward_log:
            episode_total = sum(self.episode_reward_log)
            print(f"  Episode summary: Total reward={episode_total:.3f}, "
                  f"Steps={len(self.episode_reward_log)}, "
                  f"Checkpoint={checkpoint_reached}, "
                  f"Avg Speed={avg_speed:.1f} km/h")
        
        # Reset all state
        self.current_checkpoint_idx = 0
        self.last_position = np.array([0.0, 0.0])
        self.last_speed = 0.0
        self.speed_history.clear()
        self.stuck_timer = 0.0
        self.episode_steps = 0
        self.frame_counter = 0
        self.last_distance_to_checkpoint = float('inf')
        self.consecutive_distance_increases = 0
        self.backward_movement_count = 0
        self.brake_hold_start_time = None
        self.brake_tap_times.clear()
        self.last_action_was_brake = False
        self.info_dict = {}
        self.episode_start_time = time.time()
        self.episode_reward_log = []
        self.learning_started = False
        self.checkpoint_times = []
        self.last_checkpoint_time = time.time()
        self.episode_speeds = []
        
        # Send reset command
        self._send_reset_command(reason=reset_reason)
        
        # Wait for game to reset
        time.sleep(0.2)
        
        # Read fresh telemetry
        max_attempts = 10
        telemetry = None
        for _ in range(max_attempts):
            telemetry = self._read_telemetry()
            if telemetry is not None:
                break
            time.sleep(0.05)
        
        if telemetry is None:
            print("⚠️ Warning: No telemetry after reset, using zeros")
            return np.zeros(6, dtype=np.float32), {}
        
        obs, pos, speed = self._get_observation(telemetry)
        self.last_position = np.array(pos)
        self.last_obs = obs
        
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action in environment"""
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self.last_obs
        pos = (0.0, 0.0)
        speed = 0.0

        for i in range(self.ACTION_FRAME_SKIP):
            obs_before = self.last_obs
            
            # Force accelerate during first 1.5s
            elapsed = time.time() - self.episode_start_time
            if elapsed < 1.5:
                action_to_send = 3
            else:
                action_to_send = action

            self._send_action(action_to_send)
            self.frame_counter += 1

            time.sleep(0.01)

            telemetry = self._read_telemetry()
            if telemetry is None:
                continue

            obs, pos, speed = self._get_observation(telemetry)
            self.last_position = np.array(pos)
            self.last_speed = speed

            # Check checkpoint progression with SPEED TRACKING
            x, z = pos
            if self.current_checkpoint_idx < len(self.checkpoints) - 1:
                current_cp = self.checkpoints[self.current_checkpoint_idx]
                next_cp = self.checkpoints[self.current_checkpoint_idx + 1]
                
                distance_current = np.sqrt((x - current_cp.x)**2 + (z - current_cp.z)**2)
                distance_next = np.sqrt((x - next_cp.x)**2 + (z - next_cp.z)**2)
                
                if distance_next < distance_current:
                    # Base checkpoint reward
                    checkpoint_reward = self.REWARDS['checkpoint_reached']
                    
                    # SPEED BONUS: reward for reaching checkpoint quickly
                    current_time = time.time()
                    time_since_last = current_time - self.last_checkpoint_time
                    
                    # Bonus if checkpoint reached quickly (under 10 seconds is excellent)
                    if time_since_last < 10.0:
                        speed_bonus = self.REWARDS['checkpoint_speed_bonus'] * (1.0 - time_since_last / 10.0)
                        checkpoint_reward += speed_bonus
                        print(f"  ★ CP {self.current_checkpoint_idx} in {time_since_last:.1f}s! Speed bonus: +{speed_bonus:.1f}")
                    else:
                        print(f"  ★ CP {self.current_checkpoint_idx} in {time_since_last:.1f}s")
                    
                    total_reward += checkpoint_reward
                    self.episode_reward_log.append(checkpoint_reward)
                    
                    self.checkpoint_times.append(time_since_last)
                    self.last_checkpoint_time = current_time
                    
                    old_idx = self.current_checkpoint_idx
                    self.current_checkpoint_idx += 1
                    self.last_distance_to_checkpoint = float('inf')
                    self.backward_movement_count = 0
                    self.brake_hold_start_time = None
                    self.brake_tap_times.clear()
                    self.last_action_was_brake = False
                    print(f"  ★ Checkpoint {old_idx} → {self.current_checkpoint_idx}")

            # Compute per-frame reward
            reward, term = self._compute_reward(telemetry, obs, action_to_send, obs_before)
            total_reward += reward
            self.last_obs = obs

            if term:
                terminated = True
                break

        # Mark learning_started flag
        if not self.learning_started and (time.time() - self.episode_start_time) >= 1.5:
            self.learning_started = True

        info = {
            'checkpoint': self.current_checkpoint_idx,
            'position': pos,
            'speed': speed,
            'avg_speed': np.mean(self.episode_speeds) if self.episode_speeds else 0.0,
            'learning_started': self.learning_started,
            **self.info_dict
        }

        total_reward = np.clip(total_reward, -10.0, 200.0)

        return self.last_obs, float(total_reward), bool(terminated), truncated, info
    
    def close(self):
        """Close environment"""
        if self.socket:
            self.socket.close()
    
    def save_checkpoint(self, filepath: str):
        """Save current state for resuming"""
        state = {
            'current_checkpoint_idx': self.current_checkpoint_idx,
            'episode_steps': self.episode_steps,
            'frame_counter': self.frame_counter
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_checkpoint(self, filepath: str):
        """Load saved state"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            self.current_checkpoint_idx = state.get('current_checkpoint_idx', 0)
            self.episode_steps = state.get('episode_steps', 0)
            self.frame_counter = state.get('frame_counter', 0)