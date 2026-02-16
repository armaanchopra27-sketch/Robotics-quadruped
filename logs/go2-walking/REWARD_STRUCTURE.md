# Go2-Walking Model Reward Structure

**Model Iterations**: 0-1000  
**Training Period**: February 2026  
**Command System**: Discrete commands (6 total) + Dual commands (direction + speed)

---

## Reward Configuration

### Active Rewards (Iterations 0-1000)

| Reward Function | Weight | Formula | Purpose |
|----------------|--------|---------|---------|
| **Command Tracking** | +3.0 | `exp(-‖v_actual - v_commanded‖)` | Primary signal: Follow commanded velocity |
| **Height Consistency** | +5.0 → +2.0* | `exp(-5.0 × (z - 0.3)²)` | Maintain stable torso height at 0.3m |
| **Sine Gait** | +3.0 | `Σ exp(-(joint_vel - sin(phase))²)` | Encourage natural periodic leg motion |
| **Upright Posture** | +1.0 | `exp(-(pitch² + roll²))` | Stay upright, no tipping |
| **Energy Smoothness** | -0.5 → 0.0* | `-Σ(Δaction²)` | Penalize jerky movements (disabled at iteration 900+) |
| **Constant Velocity** | +2.0 | `exp(-2.0 × ‖speed_error‖)` if speed>0.2 | Maintain consistent speed |
| **Death Penalty** | -20.0 | `-20.0` on termination | Strong negative for falling |

**Forward Motion** (DISABLED at iteration 200+): Weight set to 0.0 due to conflict with backward/turn commands

\* Height consistency reduced from 5.0→2.0 at iteration 900+ to prevent standing still  
\* Energy smoothness disabled (0.0) at iteration 1900+ to allow faster responses

---

## Command Evolution

### Phase 1: Single Discrete Commands (Iterations 0-900)
- Forward only
- 50% forward, 25% turn left, 25% turn right
- Forward motion reward caused backward movement issues

### Phase 2: 6 Discrete Commands (Iterations 900-1000)
- Forward, Backward, Turn Left, Turn Right, Strafe Left, Strafe Right
- Commands sampled randomly every 5 seconds
- Forward motion reward **disabled** (set to 0.0)

### Phase 3: Dual Commands (Iterations 900+)
- **Direction**: 0-360° (0°=forward, 90°=right, 180°=back, 270°=left)
- **Speed**: 0.3-1.5 m/s
- Direction biased: 50% forward-ish, 20% right-ish, 20% left-ish, 10% backward-ish
- Commands sampled every 5 seconds

---

## Hyperparameters

```python
Learning Rate: 0.001
PPO Clip: 0.2
GAE Lambda: 0.95
Discount Gamma: 0.99
Batch Size: 24 steps × 512 envs = 12,288
PPO Epochs: 5
Entropy Coef: 0.01
Max Grad Norm: 1.0
```

---

## Network Architecture

### Actor (Policy Network)
```
Input: 48 observations
→ Linear(512) + ELU
→ Linear(256) + ELU
→ Linear(128) + ELU
→ Linear(12) outputs (joint actions)
```

### Critic (Value Network)
```
Input: 48 observations
→ Linear(512) + ELU
→ Linear(256) + ELU
→ Linear(128) + ELU
→ Linear(1) output (state value)
```

---

## Observation Space (48 dimensions)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Joint positions | 12 | Current position of all 12 motors |
| Joint velocities | 12 | Current velocity of all 12 motors |
| Base orientation | 3 | Roll, pitch, yaw (Euler angles) |
| Base angular velocity | 3 | Angular velocity in body frame |
| Projected gravity | 3 | Gravity vector in body frame (IMU) |
| Commands | 5 | [lin_vel_x, lin_vel_y, ang_vel, height, jump] |
| Last actions | 12 | Previous timestep's actions (latency simulation) |

**Note**: Only 48 dims used; last_actions not included in observation

---

## Action Space (12 dimensions)

- **Output**: 12 continuous joint torques (normalized -1 to +1)
- **Clipping**: Actions clipped to [-5, 5] before scaling
- **Scaling**: Multiplied by action_scale and added to default positions
- **Control**: PD controller maps to actual motor torques

---

## Training Performance Milestones

| Iteration | Achievement | Notes |
|-----------|-------------|-------|
| 0-100 | Random flailing → Standing | Initial learning |
| 200 | Disabled forward_motion reward | Removed bias preventing backward movement |
| 500 | Stable forward walking | Consistent gait emerged |
| 900 | Added strafe commands | 66% reward increase in 37 iterations |
| 950 | Policy collapse | Learning rate too high, recovered by 1000 |
| 1000 | 6-command mastery | Executes all discrete commands well |

---

## Known Issues & Solutions

### Issue 1: Forward Motion Bias (Iteration 0-200)
- **Problem**: Robot wouldn't go backward
- **Cause**: forward_motion reward (+1.0 × velocity_x) conflicted with backward commands
- **Solution**: Set forward_motion weight to 0.0

### Issue 2: Standing Still (Iteration ~950)
- **Problem**: Robot stopped moving despite commands
- **Cause**: height_consistency reward too high (5.0)
- **Solution**: Reduced to 2.0, increased strictness to -10.0

### Issue 3: Policy Collapse (Iteration 950)
- **Problem**: Rewards suddenly dropped, behavior degraded
- **Cause**: Learning rate too high, value function divergence
- **Solution**: Natural recovery by iteration 1000, consider lower LR for future

---

## Checkpoint Files

All checkpoints saved in `logs/go2-walking/`:

- `model_0.pt` - Random initialization
- `model_100.pt` - Early learning
- `model_200.pt` - Forward motion disabled
- `model_300-800.pt` - Progressive improvement
- `model_900.pt` - **Recommended for dual-command transfer**
- `model_1000.pt` - Final 6-command model

---

## Transfer Learning Notes

**Best checkpoint for dual commands**: `model_900.pt`
- Robot can walk smoothly in all directions
- Basic command-following established
- Not overfit to discrete commands yet
- Transfer to continuous dual commands requires ~1000-2000 more iterations

**Avoid**: `model_1000.pt` showed policy instability during dual-command transfer

---

## Biological Parallels (For Science Fair Paper)

| RL Component | Brain Structure | Function Parallel |
|--------------|----------------|-------------------|
| Reward signals | Dopamine (VTA) | Reinforcement learning signal |
| Actor-critic | Basal ganglia | Action selection + value estimation |
| Policy network | Motor cortex (M1) | Movement execution |
| Value network | Prefrontal cortex | Outcome prediction |
| Sine gait reward | Central pattern generators | Rhythmic motor programs |
| PPO algorithm | Synaptic plasticity | Hebbian learning ("fire together, wire together") |
| Parallel envs (512) | Sleep/memory replay | Offline learning from experience |
| Command tracking | Parietal cortex | Goal-directed movement |

---

**Generated**: February 16, 2026  
**Model Path**: `logs/go2-walking/`  
**Next Phase**: Train vision network using frozen locomotion policy from checkpoint 900
