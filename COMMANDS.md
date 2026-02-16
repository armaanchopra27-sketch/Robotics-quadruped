# Go2 Quadruped Training Commands Reference

Complete reference for training and visualizing the Go2 quadruped robot with reinforcement learning.

---

## Training Commands

### Basic Training
```bash
# Start training from scratch
.\train.bat

# Train with custom number of iterations
.\train.bat --steps 5000

# Train with more parallel environments
.\train.bat --envs 8192

# Train without Weights & Biases logging
.\train.bat --no_wandb
```

### Resume from Checkpoint
```bash
# Resume from specific checkpoint iteration

.\train.bat --checkpoint 900
# Resume from latest checkpoint
.\train.bat --latest

# Resume and train for more iterations
.\train.bat --checkpoint 900 --steps 5000
```

### Advanced Training
```bash
# Custom experiment name
.\train.bat --exp_name my_experiment

# Use CPU instead of GPU (slow!)
.\train.bat --device cpu

# Full training run with all options
.\train.bat --exp_name go2_strafe --steps 10000 --envs 4096 --checkpoint 900
```

---

## Vision-Based Food Seeking (Hierarchical RL)

### Train Vision Policy

Train a high-level vision network that sees food and decides commands. The
locomotion network (trained earlier) executes the commands.

```bash
# Train vision policy using checkpoint 900 locomotion
.\train_vision.bat --locomotion_checkpoint 900

# Use fewer environments (vision training uses more memory)
.\train_vision.bat --locomotion_checkpoint 900 --num_envs 512

# Train with full vision model (slower but better)
.\train_vision.bat --locomotion_checkpoint 900 --vision_model full

# Custom experiment name
.\train_vision.bat --exp_name my_vision_exp --locomotion_checkpoint 900
```

### Hierarchical System Architecture

```
Camera (160x120 RGB)  ← Reduced for 6GB VRAM
      ↓
Vision Network (CNN)  ← TRAINS HERE
      ↓
Discrete Command (6 options)
      ↓
Locomotion Network    ← FROZEN (already trained)
      ↓
Joint Actions (12 DOF)
```

This mimics the brain:
- **Visual Cortex** → Vision CNN processes camera
- **Prefrontal Cortex** → Decision making (command selection)
- **Motor Cortex** → Locomotion network (movement execution)

---

## Visualization Commands

### Visualize Locomotion Policy
```bash
# Visualize latest checkpoint
.\visualize.bat

# Visualize specific checkpoint
.\visualize.bat --checkpoint 900

# Run multiple episodes
.\visualize.bat --checkpoint 900 --episodes 10

# Visualize without rendering (for metrics only)
.\visualize.bat --checkpoint 900 --headless
```

### Visualize Vision Policy (Food Seeking)
```bash
# Visualize latest vision checkpoint
.\visualize_vision.bat --latest

# Visualize specific vision checkpoint with locomotion checkpoint 1000
.\visualize_vision.bat --checkpoint 500 --locomotion_checkpoint 1000

# Run multiple episodes
.\visualize_vision.bat --checkpoint 1000 --episodes 5

# Custom experiment name
.\visualize_vision.bat --latest --exp_name my_vision_exp
```

**What you'll see:**
- Robot navigating to find and collect food items (green spheres)
- Command selection printed in real-time (FORWARD, TURN LEFT, etc.)
- Food collection notifications when robot reaches food
- Episode statistics: steps, rewards, food collected

**Commands displayed:**
- 🔼 FORWARD - Move straight ahead (1.0 m/s)
- 🔽 BACKWARD - Move backward (-0.5 m/s)
- ↺ TURN LEFT - Rotate counter-clockwise (0.6 rad/s)
- ↻ TURN RIGHT - Rotate clockwise (-0.6 rad/s)
- ⬅ STRAFE LEFT - Sidestep left (0.5 m/s)
- ➡ STRAFE RIGHT - Sidestep right (-0.5 m/s)

---

## Robot Discrete Commands

The robot is trained to follow 6 discrete movement commands that change every 5 seconds:

### Movement Commands

| Command | Description | Velocity Vector |
|---------|-------------|-----------------|
| **Forward** | Move straight ahead | `[1.0, 0.0, 0.0]` (x, y, yaw) |
| **Backward** | Move straight back | `[-0.5, 0.0, 0.0]` |
| **Turn Left** | Rotate counter-clockwise | `[0.0, 0.0, 0.6]` |
| **Turn Right** | Rotate clockwise | `[0.0, 0.0, -0.6]` |
| **Strafe Left** | Sidestep to the left | `[0.0, 0.5, 0.0]` |
| **Strafe Right** | Sidestep to the right | `[0.0, -0.5, 0.0]` |

### Command Format
Each command is converted to a 5D vector: `[lin_vel_x, lin_vel_y, ang_vel, height, unused]`
- **lin_vel_x**: Forward/backward velocity (m/s)
- **lin_vel_y**: Left/right velocity (m/s) - used for strafing
- **ang_vel**: Yaw angular velocity (rad/s)
- **height**: Target body height (0.3m default)
- **unused**: Reserved for future use

---

## Reward Functions

The robot learns through 7 reward signals:

### Active Rewards (weights > 0)
1. **command_tracking** (weight: 3.0)
   - Rewards following the commanded velocity
   - Penalizes deviation from desired motion

2. **height_consistency** (weight: 2.0)
   - Encourages maintaining stable body height
   - Prevents bobbing up and down

3. **sine_gait** (weight: 3.0)
   - Rewards smooth, periodic leg motion
   - Encourages natural walking pattern

4. **upright_posture** (weight: 1.0)
   - Keeps the robot upright
   - Penalizes excessive tilting

### Negative Rewards (penalties)
5. **energy_smoothness** (weight: -0.1)
   - Penalizes jerky movements
   - Encourages smooth actions

6. **death_penalty** (weight: -20.0)
   - Large penalty for falling over
   - Triggers episode termination

### Disabled Rewards
7. **forward_motion** (weight: 0.0)
   - Previously caused bias against backward movement
   - Set to 0.0 to allow full command following

---

## File Structure

### Main Scripts
- `train.py` - Main training entry point
- `visualize.py` - Policy visualization script
- `train.bat` - Windows training launcher
- `visualize.bat` - Windows visualization launcher

### Source Files
- `src/go2_env.py` - Robot environment (physics, rewards, commands)
- `src/go2_train.py` - Training logic and configuration
- `src/go2_eval.py` - Evaluation utilities

### Configuration
- Environments: 4096 parallel simulations
- Episode length: 20 seconds (1000 steps at 50Hz)
- Command interval: 5 seconds (250 steps)
- Save interval: Every 100 iterations
- Network: Actor-Critic with [512, 256, 128] hidden layers

### Checkpoints
- Location: `logs/go2-walking/`
- Format: `model_{iteration}.pt`
- Contains: Model weights, optimizer state, iteration number
- Fixed bug: Checkpoints now save correct iteration numbers!

---

## Example Workflows

### Train from Scratch
```bash
# Start new training run
.\train.bat --exp_name go2_with_strafe --steps 10000
```

### Continue Training
```bash
# Resume from checkpoint 900, train to 2000
.\train.bat --checkpoint 900 --steps 2000
```

### Evaluate Performance
```bash
# Visualize checkpoint 900 for 20 episodes
.\visualize.bat --checkpoint 900 --episodes 20
```

### Debug Training
```bash
# Short run without W&B for testing
.\train.bat --steps 10 --no_wandb
```

---

## Performance Metrics

### Training Progress Indicators
- **Mean reward**: Should increase over time (0 → 150+)
- **Episode length**: Should approach 1000 steps (max)
- **Command tracking reward**: Should increase (0 → 2.5+)
- **Death penalty**: Should decrease (frequent → rare)

### Good Checkpoint Indicators
- Reward > 150
- Episode length > 950 steps
- All movement commands executed correctly
- Robot maintains stable gait

### Policy Collapse Indicators
- Sudden reward drop
- Robot stops moving
- Only energy_smoothness reward remains positive
- Episode length drops below 100 steps

---

## Troubleshooting

### Training starts from scratch instead of resuming
**Solution**: Fixed! Checkpoints now store correct iteration numbers. Run `fix_checkpoints.py` if you have old checkpoints.

### Robot won't move backward
**Solution**: Set `forward_motion` reward weight to 0.0 (already done in current config)

### Robot falls over frequently
**Solutions**:
- Increase `upright_posture` weight
- Increase `death_penalty` magnitude
- Check if commands are too aggressive

### Training is slow
**Solutions**:
- Reduce number of environments: `--envs 2048`
- Disable camera/lidar if not needed
- Use GPU (default)

### W&B sync issues
**Solution**: Use offline mode: `--no_wandb`

---

## Science Fair Documentation

### Key Metrics to Track
1. Learning curve (reward over iterations)
2. Command success rate per command type
3. Episode length progression
4. Reward component breakdown
5. Policy robustness (variance in performance)

### Biological Analogies
- **PPO Algorithm** ↔ Dopamine-based learning (basal ganglia)
- **Reward signals** ↔ Dopamine release (reward prediction error)
- **Policy network** ↔ Motor cortex (action generation)
- **Value network** ↔ Prefrontal cortex (state evaluation)
- **Parallel envs** ↔ Memory replay during sleep

### Experimental Comparisons
1. **With/without forward_motion reward** - Shows reward shaping effects
2. **Different command sets** - Tests generalization
3. **Checkpoint progression** - Demonstrates incremental learning

---

## Additional Resources

### Genesis Simulator
- Physics engine for robotics
- Supports 4096+ parallel environments
- GPU-accelerated simulation

### RSL-RL Library
- PPO implementation from ETH Zurich
- Actor-critic architecture
- Designed for legged robots

### Weights & Biases
- Experiment tracking
- Real-time metrics visualization
- Project: `go2_locomotion`

---

## Quick Start: Two-Stage Brain Training

### Stage 1: Motor Cortex (Locomotion) - DONE ✅
```bash
# You already trained this to checkpoint 900+
.\train.bat --checkpoint 900 --steps 2000
```

Your robot now has a trained "motor cortex" that can execute 6 movement commands.

### Stage 2: Visual Cortex (Food Seeking) - START HERE 🧠
```bash
# Train vision network to see food and navigate
.\train_vision.bat --locomotion_checkpoint 900
```

This creates a hierarchical system exactly like your brain:
1. **Eyes → Visual cortex** (Camera → CNN)
2. **Decision making** (Prefrontal cortex → Command selection)
3. **Movement execution** (Motor cortex → Locomotion network)

The locomotion network is **frozen** - only the vision network learns. This is
transfer learning, just like how you use existing motor skills to do new tasks!

### Science Fair Impact

This demonstrates:
- **Hierarchical learning** - How the brain separates vision from movement
- **Transfer learning** - Reusing trained motor skills for new tasks
- **Emergent behavior** - Food seeking emerges from simple rewards
- **Biological inspiration** - Same architecture as mammalian brain

**Metrics to track:**
- Food collected per episode
- Path efficiency (distance traveled vs. straight line)
- Command usage patterns (which commands used when)
- Learning speed (iterations to first food collection)

---

**Last Updated**: February 14, 2026  
**Current Version**: Checkpoint 900+ with strafing support + Hierarchical vision system
