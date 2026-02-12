# Go2 Quadruped Training - Quick Start

## Checkpoint Location
Checkpoints are saved in: `logs/{exp_name}/model_{iteration}.pt`
- Default experiment name: `go2-walking`
- Example: `logs/go2-walking/model_1000.pt`, `logs/go2-walking/model_2000.pt`, etc.

## Visualize Trained Policy

View your trained robot in action:

```bash
# Visualize latest checkpoint (5 episodes)
.\visualize.bat --latest

# Visualize specific checkpoint
.\visualize.bat --checkpoint 5000

# Run 10 episodes instead of 5
.\visualize.bat --latest --episodes 10
```

## Training Commands

### Start training from scratch
```bash
.\train.bat
```

### Resume from latest checkpoint
```bash
.\train.bat --latest
```

### Resume from specific checkpoint
```bash
.\train.bat --checkpoint 5000
```

### Custom training configuration
```bash
.\train.bat --steps 50000 --envs 8192
```

### Training without W&B logging
```bash
.\train.bat --no_wandb
```

## All Arguments

- `--latest` - Resume from latest checkpoint
- `--checkpoint N` - Resume from specific checkpoint iteration N
- `--steps N` - Number of training iterations (default: 10000)
- `--envs N` - Number of parallel environments (default: 4096)
- `--device cuda:0` - Device to use (default: auto-detect GPU, falls back to CPU)
- `--no_wandb` - Disable W&B logging (enabled by default)
- `--exp_name NAME` - Experiment name (default: go2-walking)

## Features

✅ **Checkpoints saved every 1,000 steps** in `logs/{exp_name}/model_{iteration}.pt`
✅ **Visualization every 1,000 steps** (shows 5 episodes with current policy)
✅ W&B logging enabled by default
✅ 6 reward components tracked:
  - forward_motion
  - height_consistency
  - sine_gait
  - upright_posture
  - energy_smoothness
  - death_penalty
