# Hybrid Double DQN for TrackMania Nations Forever (TMNF)

A real-time Reinforcement Learning system that trains a Double DQN agent to autonomously drive in **TrackMania Nations Forever (TMNF)** using live telemetry via socket communication.

This project implements a **Hybrid Replay Strategy** combining elite experience replay and recent policy replay to prevent learning stagnation while maintaining performance quality.

---

## System Overview

```
TMNF Game (Real-time)
        │
        ▼
TMInterface Socket (TCP Telemetry)
        │
        ▼
TMNFEnvironment (Gymnasium Wrapper)
        │
        ▼
Hybrid DQN Agent
   ├── Online Q-Network
   ├── Target Q-Network
   └── Hybrid Replay Buffer
        │
        ▼
Training Loop (Dynamic Elite Sampling)
```

This is a real-time online RL system interacting directly with the game engine — not a simulator.

---

# 1. Environment Architecture

### Class: `TMNFEnvironment`

Implements:

```python
gym.Env
```

---

## Observation Space (6D State Vector)

$$s_t = [x, z, v, c, dx, dz]$$

Where:

* `x, z` → Position
* `v` → Speed (km/h)
* `c` → Current checkpoint index
* `dx, dz` → Relative vector to target checkpoint

The observation space is defined using a bounded `Box` space.

---

## Action Space (Discrete 5)

| Action | Meaning    |
| ------ | ---------- |
| 0      | No-op      |
| 1      | Left       |
| 2      | Right      |
| 3      | Accelerate |
| 4      | Brake      |

---

## Reward Function (Speed-Focused)

The reward function is multi-component:

$$R_t = R_{\text{time}} + R_{\text{speed}} + R_{\text{distance}} + R_{\text{checkpoint}} + R_{\text{finish}} + R_{\text{penalties}}$$

---

### 1. Time Penalty

$$R_{\text{time}} = -0.01$$

Encourages faster completion.

---

### 2. Speed Bonus

$$R_{\text{speed}} = 0.002 \cdot \min\left(\frac{v}{200}, 1\right)$$

Rewards maintaining high speed.

---

### 3. Distance Progress Reward

$$R_{\text{distance}} = \text{clip}(0.05 \cdot (d_{t-1} - d_t), -0.1, 0.1)$$

Smooth shaping toward checkpoints.

---

### 4. Checkpoint Reward

Base reward:

$$+5.0$$

Fast arrival bonus:

$$3.0 \cdot \left(1 - \frac{t_{\text{cp}}}{10}\right)$$

---

### 5. Finish Reward

$$R_{\text{finish}} = 100 + 50 \cdot \frac{v}{200}$$

Strong terminal incentive.

---

### 6. Penalties

* Collision: -2
* Wall climb: -5
* Stuck: -2
* Brake spam: -10

---

## Termination Conditions

Episodes terminate when:

* Finish line crossed
* Timeout
* Max steps reached
* Wall climb detected
* Stuck condition triggered
* Excessive brake spam

---

# 2. DQN Network Architecture

Defined in `DQNNetwork`.

## Structure

```
Input (6)
   ↓
Linear(6 → 128)
   ↓ ReLU
Linear(128 → 128)
   ↓ ReLU
Linear(128 → 64)
   ↓ ReLU
Linear(64 → 5)
```

Hidden Layers:

* 128
* 128
* 64

Output:

$$Q(s,a) \in \mathbb{R}^5$$

---

## Initialization

Orthogonal initialization:

$$W_i \sim \text{Orthogonal}(\sqrt{2})$$

Bias initialized to 0.

This improves gradient flow stability.

---

# 3. Double DQN Formulation

Inside the training step:

Action selection:

$$a' = \arg\max_a Q_{\theta}(s', a)$$

Target calculation:

$$y = r + \gamma Q_{\theta^-}(s', a')$$

Where:

* $Q_{\theta}$ = Online network
* $Q_{\theta^-}$ = Target network
* $\gamma = 0.99$

Loss function:

$$\mathcal{L} = \text{SmoothL1}(Q_{\theta}(s,a), y)$$

Gradient clipping:

$$|\nabla| \le 1.0$$

---

# 4. Hybrid Replay Buffer Architecture

Core innovation of this project.

Two buffers are maintained:

## Elite Buffer

* Stores top 20% episodes (ranked by performance score)
* Preserves high-performing trajectories

## Recent Buffer

* Stores latest 15,000 transitions
* Reflects current policy distribution

---

## Performance Score

Episodes are ranked using:

$$S = 1000 \cdot c + 2 \cdot v_{avg} + (3000 - t_{episode})$$

Where:
- $c$ = number of checkpoints reached
- $v_{avg}$ = average speed during episode
- $t_{episode}$ = number of steps taken

Higher score → higher elite priority.

---

## Hybrid Sampling Strategy

Batch composition:

$$B = B_{\text{elite}} \cup B_{\text{recent}}$$

Sampling ratio evolves during training:

* Early: 30% elite
* Mid: 50% elite
* Late: 60% elite

This creates a curriculum-like progression.

---

# 5. Training Loop Structure

For each episode:

1. Reset environment
2. Collect trajectory
3. Compute episode metrics
4. Insert into hybrid buffer
5. Perform 5 gradient updates
6. Periodic evaluation
7. Save checkpoints

---

## Exploration Strategy

Epsilon-greedy:

$$\epsilon_t = \max(0.10, \epsilon_{t-1} \cdot 0.9995)$$

When loading a pretrained model, epsilon is reset to 0.3 to encourage renewed exploration.

---

# 6. Target Network Strategy

Hard update every:

$$1000 \text{ training steps}$$

This stabilizes Q-value bootstrapping.

---

# 7. Complete Data Flow

```
State s_t
   ↓
Q-network
   ↓
Action a_t (ε-greedy)
   ↓
Environment
   ↓
(s', r, done)
   ↓
Store transition
   ↓
Hybrid Replay Buffer
   ↓
Sample mixed batch
   ↓
Double DQN update
   ↓
Update θ
   ↓
Periodic copy to θ⁻
```

---

# 8. Why This Architecture Works

This system addresses three classic DQN weaknesses:

### 1. Overestimation Bias

Solved via Double DQN.

### 2. Catastrophic Forgetting

Elite replay preserves high-quality trajectories.

### 3. Policy Drift

Recent buffer maintains distribution alignment with the current policy.

---

# 9. Computational Complexity

Per update:

$$O(B \cdot 128^2)$$

Where $B = 64$.

Replay memory limits:

* Elite transitions ≤ 50,000
* Recent transitions ≤ 15,000

Total ≤ 65,000 transitions.

---
