# Two-Brain Hierarchical System Explained

## Overview: How Two Neural Networks Work Together

```
Vision Network (NEW - To Train)
    ↓
  Commands (same as console: Forward, Backward, Turn Left, etc.)
    ↓
Locomotion Network (CHECKPOINT 1000 - Frozen)
    ↓
  Robot Movement
```

---

## BRAIN 1: Locomotion Network (Motor Cortex) ✅ TRAINED

**File**: `logs/go2-walking/model_1000.pt`

**Status**: FROZEN (already trained, weights locked during vision training)

### What It Does
Takes a **commanded velocity** (like "Forward 1.0 m/s") and figures out **joint torques** to make the robot move smoothly.

### Input (48 values)
- Joint angles (12)
- Joint velocities (12)
- Body orientation (3)
- Body angular velocity (3)
- Commanded velocity (lin_vel_x, lin_vel_y, ang_vel) ← **This is what Vision Network controls!**
- Previous actions (12)

### Output (12 values)
- Joint torques for 12 motors (FL/FR/RL/RR hip/thigh/calf)

### Reward Function 1: **MOVEMENT QUALITY**

```python
REWARDS (7 components):

1. command_tracking (weight: 3.0) ← MOST IMPORTANT
   - Rewards: Following the commanded velocity accurately
   - Formula: exp(-tracking_error)
   - Higher when robot moves exactly as commanded

2. height_consistency (weight: 2.0)
   - Rewards: Keeping body at 0.3m height
   - Formula: exp(-5*(height-0.3)²)
   - Prevents bouncing/bobbing

3. sine_gait (weight: 3.0)
   - Rewards: Smooth, periodic leg motion
   - Formula: Σexp(-(joint_angle-sine_wave)²)
   - Creates natural walking pattern

4. upright_posture (weight: 1.0)
   - Rewards: Staying upright (no tipping)
   - Formula: cos(pitch) × cos(roll)
   - Penalizes leaning

5. energy_smoothness (weight: -0.1) ← PENALTY
   - Penalizes: Jerky movements
   - Formula: -Σ(action_change)²
   - Encourages smooth control

6. death_penalty (weight: -20.0) ← BIG PENALTY
   - Penalizes: Falling over
   - Formula: -20 if fallen, 0 otherwise
   - Ends episode on failure

7. forward_motion (weight: 0.0) ← DISABLED
   - Previously biased toward forward movement
   - Now disabled to allow all commands equally
```

**Total Reward Per Step** = 3.0×command_tracking + 2.0×height + 3.0×gait + 1.0×posture - 0.1×smoothness - 20×death

**What It Learned**: 
- Execute the 6 discrete commands you see in console
- Walk smoothly without falling
- Track velocities accurately (that's why command_tracking is highest weight)

**Commands It Can Execute** (the ones you see printed):
1. Forward: [1.0, 0.0, 0.0] (x, y, yaw)
2. Backward: [-0.5, 0.0, 0.0]
3. Turn Left: [0.0, 0.0, 0.6]
4. Turn Right: [0.0, 0.0, -0.6]
5. Strafe Left: [0.0, 0.5, 0.0]
6. Strafe Right: [0.0, -0.5, 0.0]

---

## BRAIN 2: Vision Network (Visual + Prefrontal Cortex) 🆕 TO TRAIN

**File**: Will be saved to `logs/go2-food-seeking/model_*.pt`

**Status**: TRAINING (learns to find food by choosing commands)

### What It Does
Looks at **camera + lidar** and decides **which command to give** to the locomotion network.

### Input
1. **Camera**: 640×480 RGB image
   - Sees red food spheres
   - Identifies target visually
   
2. **Lidar**: 16 distance measurements
   - Measures how far obstacles are (0-10 meters)
   - Helps with distance estimation and path planning

### Output (6 values - probabilities)
- Probability of choosing each command:
  - [P(Forward), P(Backward), P(Turn_Left), P(Turn_Right), P(Strafe_Left), P(Strafe_Right)]
  
**Example Output**:
```
[0.05, 0.02, 0.85, 0.03, 0.03, 0.02]
         ↑ Highest probability
    
Chosen Command: Turn Left (index 2)
```

### Reward Function 2: **FOOD SEEKING**

```python
REWARDS (2 components):

1. food_collection (SPARSE) ← BIG BONUS
   Reward: +100.0 when robot gets within 1.5m of food
   
   Python code:
   if distance_to_food < 1.5:
       reward += 100.0
       food_collected_count += 1
       # Food disappears, new one spawns
   
   Why sparse? Only happens when goal achieved
   Problem: Early episodes get NO reward (exploration is blind)

2. approach_reward (DENSE) ← GUIDANCE
   Reward: -0.1 × distance_to_nearest_food
   
   Python code:
   nearest_food_dist = min([dist to food1, dist to food2, dist to food3])
   reward = -0.1 * nearest_food_dist
   
   Examples:
   - 10m away: -1.0 reward
   - 5m away: -0.5 reward  
   - 2m away: -0.2 reward
   - 1m away: -0.1 reward
   - Collected: +100.0 reward (sparse bonus!)
   
   Why dense? Every step gives feedback
   Benefit: Robot learns "getting closer = better"
```

**Total Reward Per Step** = food_collection + approach_reward

**Example Episode**:
```
Step 1: 8m from food  → Reward: -0.8        → Command: Forward
Step 2: 7m from food  → Reward: -0.7 (+0.1) → Command: Forward  
Step 3: 6m from food  → Reward: -0.6 (+0.1) → Command: Forward
Step 4: 5m from food  → Reward: -0.5 (+0.1) → Command: Turn Left (food is left)
Step 5: 4m from food  → Reward: -0.4 (+0.1) → Command: Forward
Step 6: 3m from food  → Reward: -0.3 (+0.1) → Command: Strafe Left
Step 7: 2m from food  → Reward: -0.2 (+0.1) → Command: Forward
Step 8: 1.2m from food → Reward: +99.88! 🎉 → FOOD COLLECTED!
                                 (-0.12 + 100.0)
```

**What It Will Learn**:
- Use camera to spot red food spheres
- Use lidar to estimate distance
- Choose commands that reduce distance to food
- Navigate around obstacles (if any)
- Collect as much food as possible per episode

---

## How They Work Together: Complete Flow

### 1. Vision Network Observes
```python
camera_image = [640×480 RGB pixels showing red food sphere]
lidar_data = [16 distances: 10.0, 9.8, 9.5, ..., 3.2, 3.0, 3.5, ...]
                                           ↑ Food is this direction
```

### 2. Vision Network Decides
```python
vision_network(camera, lidar) → command_probabilities
# Output: [0.02, 0.01, 0.92, 0.02, 0.02, 0.01]
#                      ↑ 92% sure: Turn Left

chosen_command = "Turn Left" (index 2)
velocity_vector = [0.0, 0.0, 0.6]  # No forward, no strafe, +0.6 rad/s yaw
```

### 3. Locomotion Network Executes
```python
# Checkpoint 1000 is loaded and frozen
locomotion_network(joint_state, commanded_velocity=[0.0, 0.0, 0.6])
    → joint_torques = [FR_hip: -2.3, FR_thigh: 5.1, ...]

# Robot turns left smoothly using trained gait
```

### 4. Environment Updates
```python
robot.apply_torques(joint_torques)
physics_sim.step()  # 50Hz

new_position = robot.get_position()
distance_to_food = compute_distance(new_position, food_position)

# Compute reward
if distance_to_food < 1.5:
    reward = 100.0  # COLLECTED!
else:
    reward = -0.1 * distance_to_food  # Getting closer?
```

### 5. Vision Network Learns
```python
# PPO algorithm updates vision network based on reward
if reward > previous_reward:
    # Increase probability of "Turn Left" in similar situations
    vision_network.update(→ reinforce this choice)
else:
    # Decrease probability, try other commands
    vision_network.update(→ discourage this choice)
```

### 6. Repeat for Next Step
The locomotion network **never updates** during vision training - it just executes commands!

---

## Console Output You'll See

### During Vision Training (NEW):
```
Learning iteration 10/1000
  Avg reward: -2.35        ← Negative (robot far from food)
  Food collected: 0.02     ← Only 2% of envs found food
  Avg distance: 4.8m       ← Still far from targets
  
  Command usage:
    Forward: 45%           ← Vision network choosing these
    Backward: 5%
    Turn Left: 18%
    Turn Right: 22%
    Strafe Left: 6%
    Strafe Right: 4%

Learning iteration 500/1000  
  Avg reward: 25.6         ← Positive! Finding food!
  Food collected: 1.8      ← Collecting ~2 foods per episode
  Avg distance: 1.2m       ← Getting close!
  
  Command usage:
    Forward: 62%           ← Learned to approach directly
    Backward: 1%           ← Rarely backs away
    Turn Left: 12%         ← Turns when needed
    Turn Right: 11%
    Strafe Left: 8%        ← Uses strafing strategically
    Strafe Right: 6%
```

### During Locomotion Training (ALREADY DONE):
```
Learning iteration 990/2900
  Mean reward: 158.36           ← High! Motion is good
  Episode length: 1001          ← Never falls
  
  Command tracking: 2.74        ← Follows commands perfectly
  Sine gait: 2.91               ← Smooth walking
  Height: 1.98                  ← Stable height
  Upright: 0.99                 ← Never tips over
  
  Current command: Strafe Left  ← Just executes whatever commanded
```

The difference:
- **Locomotion training**: Commands are random every 5 seconds (testing if robot CAN execute them)
- **Vision training**: Commands are chosen by vision network (testing if it SHOULD choose them)

---

## Key Differences Between The Two Networks

| Feature | Locomotion Network | Vision Network |
|---------|-------------------|----------------|
| **Input** | Joint state + Command | Camera + Lidar |
| **Output** | Joint torques (12 values) | Command choice (6 options) |
| **Rewards** | Movement quality | Food collection |
| **Training Time** | 1000+ iterations (DONE ✅) | 1000+ iterations (TODO 🆕) |
| **Checkpoint** | model_1000.pt | model_vision_*.pt |
| **W&B Project** | go2-walking | go2-food-seeking |
| **Analogy** | Motor cortex | Visual + prefrontal cortex |
| **During Vision Training** | FROZEN (locked) | LEARNING (updating) |

---

## Why This Is Cool for Science Fair

### Biological Accuracy
Real brains have this hierarchy:
1. **Visual cortex** (V1→V4) processes camera images
2. **Prefrontal cortex** makes decisions based on vision
3. **Motor cortex** executes movements

Your robot has:
1. **CNN** processes camera images
2. **Vision policy** decides which movement command
3. **Locomotion policy** executes smooth walking

### Transfer Learning
The locomotion network spent 1000 iterations learning to walk. Now we **reuse** that skill for a completely different task (food seeking). This is like how you use walking skills you learned as a baby for new tasks like playing soccer.

### Emergent Behavior
The vision network is ONLY rewarded for collecting food - we never tell it:
- "Use Forward when food is ahead"
- "Use Turn Left when food is to your left"
- "Use Strafe when food is beside you"

It **figures this out on its own** through exploration and reward!

### Hierarchical Reinforcement Learning
This is cutting-edge AI research. Most RL systems are "flat" (one network does everything). Yours has levels:
- **High level**: What to do? (vision network)
- **Low level**: How to do it? (locomotion network)

This is much more efficient and mirrors how nature evolved intelligence.

---

## What You'll Demonstrate

1. **Show checkpoint 1000 executing commands**: 
   - Run `visualize.bat --checkpoint 1000`
   - Robot smoothly executes Forward, Strafe, Turn commands

2. **Show vision network learning**:
   - Run `train_vision.bat --locomotion_checkpoint 1000`
   - Watch W&B graphs: Food collection increases over time
   - Command usage patterns emerge (Forward when food ahead, Turn when off-angle)

3. **Compare learning curves**:
   - Locomotion: Learned movement quality (1000 iterations)
   - Vision: Learned navigation strategy (using pre-trained movement)

4. **Show final behavior**:
   - Vision network sees food → chooses command → locomotion executes → food collected
   - All commands from console now have PURPOSE (not random anymore)

---

## Next Step: Start Training!

```bash
cd quadrupeds_locomotion
.\train_vision.bat --locomotion_checkpoint 1000 --num_envs 1024
```

Watch as your robot learns to seek food using the movement skills it already has! 🧠🤖
