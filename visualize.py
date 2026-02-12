#!/usr/bin/env python3
"""
Visualize trained Go2 quadruped policy.

Examples:
    python visualize.py --latest
    python visualize.py --checkpoint 5000
    python visualize.py --checkpoint 10000 --episodes 10
"""

import os
import sys
import argparse
import glob
import torch

# Remove local genesis folder from import path to avoid conflicts
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

# Import genesis package (installed via pip)
import genesis as gs

# Now add src to path for go2_env
sys.path.insert(0, os.path.join(script_dir, 'src'))

from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner


def find_latest_checkpoint(log_dir):
    """Find the latest checkpoint in the log directory"""
    checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoint_files:
        return None, None
    
    iterations = []
    for ckpt in checkpoint_files:
        basename = os.path.basename(ckpt)
        try:
            iter_num = int(basename.replace("model_", "").replace(".pt", ""))
            iterations.append(iter_num)
        except ValueError:
            continue
    
    if not iterations:
        return None, None
    
    latest_iter = max(iterations)
    return os.path.join(log_dir, f"model_{latest_iter}.pt"), latest_iter


def get_cfgs():
    """Get default configurations"""
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    
    obs_cfg = {
        "num_obs": 48,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "base_height_target": 0.3,
        "jump_reward_steps": 50,
        "reward_scales": {
            "forward_motion": 1.5,
            "height_consistency": 2.0,
            "sine_gait": 3.0,
            "upright_posture": 1.0,
            "energy_smoothness": -0.1,
            "death_penalty": -20.0,
        },
    }
    
    command_cfg = {
        "num_commands": 5,
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }
    
    train_cfg = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "policy_class_name": "ActorCritic",
            "num_steps_per_env": 24,
            "max_iterations": 1,
            "save_interval": 1000,
        },
    }
    
    return env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg


def main():
    parser = argparse.ArgumentParser(description="Visualize trained Go2 policy")
    parser.add_argument("--exp_name", type=str, default="go2-walking", help="Experiment name")
    parser.add_argument("--latest", action="store_true", help="Load latest checkpoint")
    parser.add_argument("--checkpoint", type=int, default=None, help="Load specific checkpoint iteration")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (default: 5)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: auto-detect GPU)")
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Initialize Genesis
    gs.init(logging_level="warning")
    
    # Find checkpoint
    log_dir = f"logs/{args.exp_name}"
    
    if not os.path.exists(log_dir):
        print(f"❌ Error: Log directory not found: {log_dir}")
        print(f"   Make sure you've trained a model first!")
        return
    
    if args.latest:
        checkpoint_path, iteration = find_latest_checkpoint(log_dir)
        if checkpoint_path is None:
            print(f"❌ Error: No checkpoints found in {log_dir}")
            return
        print(f"📂 Loading latest checkpoint: iteration {iteration}")
    elif args.checkpoint is not None:
        checkpoint_path = os.path.join(log_dir, f"model_{args.checkpoint}.pt")
        iteration = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
            return
        print(f"📂 Loading checkpoint: iteration {iteration}")
    else:
        print("❌ Error: Must specify --latest or --checkpoint <iteration>")
        parser.print_help()
        return
    
    # Load configs
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = get_cfgs()
    
    # Create environment with viewer
    print(f"🌍 Creating visualization environment...")
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
        show_viewer=True,
    )
    
    # Set camera
    env.scene.viewer.set_camera_pose(pos=(-2.5, 0.0, 1.5), lookat=(0, 0, 0.5))
    
    # Create runner and load checkpoint
    print(f"🤖 Loading policy...")
    runner = OnPolicyRunner(env, train_cfg, None, device=args.device)
    
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    runner.alg.actor_critic.eval()
    
    print(f"\n{'='*70}")
    print(f"🎬 Running {args.episodes} episodes with policy from iteration {iteration}")
    print(f"{'='*70}\n")
    
    # Run episodes
    obs = env.get_observations()
    episodes_completed = 0
    steps = 0
    total_reward = 0.0
    
    with torch.inference_mode():
        while episodes_completed < args.episodes:
            actions = runner.alg.actor_critic.act_inference(obs)
            obs, _, rewards, dones, _ = env.step(actions, is_train=False)
            
            total_reward += rewards[0].item()
            steps += 1
            
            if dones[0]:
                episodes_completed += 1
                avg_reward = total_reward / steps
                print(f"  Episode {episodes_completed}/{args.episodes} complete - Steps: {steps}, Avg Reward: {avg_reward:.2f}")
                total_reward = 0.0
                steps = 0
    
    print(f"\n✓ Visualization complete - {args.episodes} episodes shown\n")


if __name__ == "__main__":
    main()
