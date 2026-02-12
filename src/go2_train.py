
import os
import argparse
import pickle
import shutil
import glob
import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def find_latest_checkpoint(log_dir):
    """Find the latest checkpoint in the log directory"""
    checkpoint_files = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not checkpoint_files:
        return None, None
    
    # Extract iteration numbers and find the maximum
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


def load_checkpoint(runner, checkpoint_path):
    """Load a checkpoint into the runner"""
    print(f"\n{'='*70}")
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=runner.device)
    runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
    runner.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    runner.current_learning_iteration = checkpoint['iter']
    
    print(f"   ✓ Loaded iteration: {checkpoint['iter']}")
    if 'infos' in checkpoint:
        if 'tot_time' in checkpoint['infos']:
            print(f"   ✓ Total time: {checkpoint['infos']['tot_time']:.1f}s")
        if 'mean_reward' in checkpoint['infos']:
            print(f"   ✓ Mean reward: {checkpoint['infos']['mean_reward']:.2f}")
    
    print(f"{'='*70}\n")
    return checkpoint['iter']


def get_next_run_number(project_name, base_name):
    """Get the next run number for wandb run naming (run1, run2, etc.)"""
    if not WANDB_AVAILABLE:
        return f"{base_name}_1"
    
    try:
        api = wandb.Api()  # type: ignore
        runs = api.runs(f"{api.viewer.username}/{project_name}")
        
        # Find all runs with base_name pattern
        max_num = 0
        for run in runs:
            if run.name and run.name.startswith(base_name):
                # Extract number from "base_name_N" or "base_nameN"
                name_suffix = run.name.replace(base_name, "").replace("_", "")
                try:
                    num = int(name_suffix)
                    max_num = max(max_num, num)
                except ValueError:
                    continue
        
        return f"{base_name}_{max_num + 1}"
    except:
        # If API call fails, use timestamp-based naming
        import time
        return f"{base_name}_{int(time.time()) % 10000}"


def get_train_cfg(exp_name, max_iterations):

    train_cfg_dict = {
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
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 1000,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
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
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
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
            # Core 6 rewards only
            "forward_motion": 1.5,          # +1.5 × v_x
            "height_consistency": 2.0,       # +2.0 × exp(-5(z-0.3)²)
            "sine_gait": 3.0,                # +3.0 × Σexp(-(pos-sine)²)
            "upright_posture": 1.0,          # +1.0 × cos(pitch)×cos(roll)
            "energy_smoothness": -0.1,       # -0.1 × Σ(Δaction)²
            "death_penalty": -20.0,          # -20.0 on termination
        },
    }
    command_cfg = {
        "num_commands": 5,  # [lin_vel_x, lin_vel_y, ang_vel, height, jump]
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
        # "lin_vel_x_range": [0.0, 0.0],
        # "lin_vel_y_range": [0.0, 0.0],
        # "ang_vel_range": [0.0, 0.0],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def run_visualizer(runner, env_cfg, obs_cfg, reward_cfg, command_cfg, device, iteration):
    """Run visualization with current policy for 5 episodes (~5000 steps)"""
    print(f"\n{'='*70}")
    print(f"🎬 Running visualizer at iteration {iteration} (5 episodes)")
    print(f"{'='*70}\n")
    
    # Create a single environment for visualization
    vis_env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=device,
        show_viewer=True,
    )
    
    # Set viewer camera
    vis_env.scene.viewer.set_camera_pose(pos=(-2.5, 0.0, 1.5), lookat=(0, 0, 0.5))
    
    # Run for 5 episodes (episode_length_s=20s, dt=0.02 -> 1000 steps/episode -> 5000 steps total)
    obs = vis_env.get_observations()
    runner.alg.actor_critic.eval()  # Set to eval mode
    
    episodes_completed = 0
    steps = 0
    with torch.inference_mode():
        while episodes_completed < 5:
            # Get action from current policy
            actions = runner.alg.actor_critic.act_inference(obs.to(device))
            obs, _, _, dones, _ = vis_env.step(actions, is_train=False)
            steps += 1
            if dones[0]:
                episodes_completed += 1
                print(f"  Episode {episodes_completed}/5 complete (step {steps})")
            
    runner.alg.actor_critic.train()  # Set back to train mode
    print(f"\n✓ Visualization complete - {episodes_completed} episodes, {steps} steps\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--device", type=str, default=None, help="Device to use: 'cpu' or 'cuda:0' (default: auto-detect GPU)")
    parser.add_argument("--wandb", action="store_true", default=True, help="Enable W&B logging (default: True)")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb", help="Disable W&B logging")
    parser.add_argument("--wandb_project", type=str, default="go2_locomotion", help="W&B project name")
    parser.add_argument("--latest", action="store_true", help="Load latest checkpoint")
    parser.add_argument("--checkpoint", type=int, default=None, help="Load specific checkpoint iteration")
    args = parser.parse_args()
    
    # Map to internal variable names
    args.num_envs = args.envs
    args.max_iterations = args.steps
    args.use_wandb = args.wandb
    
    # Auto-detect and set device (defaults to GPU if available)
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🚀 Using GPU: {gpu_name}\n")
        else:
            args.device = "cpu"
            print("\n" + "="*70)
            print("⚠️  WARNING: CUDA not available! Using CPU.")
            print("   Training will be MUCH slower on CPU.")
            print("="*70 + "\n")
    elif args.device == "cuda:0" or args.device.startswith("cuda"):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"\n🚀 Using GPU: {gpu_name}\n")
        else:
            print("\n" + "="*70)
            print("⚠️  ERROR: CUDA requested but not available!")
            print("   Falling back to CPU. Training will be MUCH slower.")
            print("="*70 + "\n")
            args.device = "cpu"
    else:
        print(f"\n⚠️  Training on CPU - this will be slow!\n")

    gs.init(logging_level="warning")  # type: ignore
    
    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("ERROR: wandb requested but not installed. Run: pip install wandb")
            return
        
        # Generate run name (run1, run2, etc.)
        run_name = get_next_run_number(args.wandb_project, args.exp_name)
        
        wandb.init(  # type: ignore
            project=args.wandb_project,
            name=run_name,
            config={
                "num_envs": args.num_envs,
                "max_iterations": args.max_iterations,
                "device": args.device,
                "resume_from_checkpoint": args.latest or (args.checkpoint is not None),
            }
        )
        print(f"✓ Weights & Biases logging enabled")
        print(f"   Project: {args.wandb_project}")
        print(f"   Run name: {run_name}\n")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)
    
    # Add configs to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.config.update({  # type: ignore
            "env_cfg": env_cfg,
            "obs_cfg": obs_cfg,
            "reward_cfg": reward_cfg,
            "command_cfg": command_cfg,
            "train_cfg": train_cfg,
        })

    # Handle log directory - don't delete if resuming from checkpoint
    resuming = args.latest or args.checkpoint is not None
    if not resuming:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device=args.device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)  # type: ignore

    # Load checkpoint if requested
    start_iteration = 0
    if args.latest or args.checkpoint is not None:
        if args.latest:
            ckpt_path, start_iteration = find_latest_checkpoint(log_dir)
            if ckpt_path is None:
                print(f"⚠️  No checkpoints found in {log_dir}, starting from scratch")
                start_iteration = 0
            else:
                start_iteration = load_checkpoint(runner, ckpt_path)
        elif args.checkpoint is not None:
            ckpt_path = os.path.join(log_dir, f"model_{args.checkpoint}.pt")
            if not os.path.exists(ckpt_path):
                print(f"❌ Checkpoint not found: {ckpt_path}")
                return
            start_iteration = load_checkpoint(runner, ckpt_path)
    
    # Save configs (skip if resuming to preserve original)
    if not resuming or not os.path.exists(f"{log_dir}/cfgs.pkl"):
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
            open(f"{log_dir}/cfgs.pkl", "wb"),
        )

    # Monkey-patch logging into runner for wandb and visualization
    original_log = runner.log
    def log_with_extras(locs, *log_args, **log_kwargs):
        # Call original tensorboard logging
        original_log(locs, *log_args, **log_kwargs)
        
        # Wandb logging - ONLY the 6 specific rewards
        if args.use_wandb and WANDB_AVAILABLE:
            if locs.get('rewbuffer') and len(locs['rewbuffer']) > 0:
                import statistics
                
                log_dict = {
                    "train/total_reward": statistics.mean(locs['rewbuffer']),
                }
                
                # Only log the 6 specific reward components
                if locs.get('ep_infos'):
                    wanted_rewards = [
                        'forward_motion',      # +1.5 × v_x
                        'height_consistency',  # +2.0 × exp(-5(z-0.3)²)
                        'sine_gait',          # +3.0 × Σexp(-(pos-sine)²)
                        'upright_posture',    # +1.0 × cos(pitch)×cos(roll)
                        'energy_smoothness',  # -0.1 × Σ(Δaction)²
                        'death_penalty',      # -20.0 on termination
                    ]
                    
                    for reward_name in wanted_rewards:
                        key = f'rew_{reward_name}'
                        if key in locs['ep_infos'][0]:
                            infotensor = torch.tensor([], device=runner.device)
                            for ep_info in locs['ep_infos']:
                                val = ep_info[key]
                                if not isinstance(val, torch.Tensor):
                                    val = torch.Tensor([val])
                                if len(val.shape) == 0:
                                    val = val.unsqueeze(0)
                                infotensor = torch.cat((infotensor, val.to(runner.device)))
                            log_dict[f"rewards/{reward_name}"] = torch.mean(infotensor).item()
                
                wandb.log(log_dict, step=locs['it'])  # type: ignore
        
        # Run visualizer every 1k steps (5 episodes)
        if locs['it'] % 1000 == 0 and locs['it'] > 0:
            run_visualizer(runner, env_cfg, obs_cfg, reward_cfg, command_cfg, args.device, locs['it'])
    
    runner.log = log_with_extras

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()  # type: ignore


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
