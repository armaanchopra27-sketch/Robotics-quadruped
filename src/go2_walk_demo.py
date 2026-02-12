"""
Demo to make the Go2 robot walk forward automatically.
This sets velocity commands directly - robot will attempt to walk.
"""

import torch
import genesis as gs
from go2_env import Go2Env

def main():
    # Initialize Genesis
    gs.init(backend=gs.constants.backend.cpu, logging_level="warning")
    
    print("Creating Go2 quadruped environment...")
    
    # Configuration
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "dof_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0, "kd": 0.5,
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
            "lin_vel": 2.0, "ang_vel": 0.25,
            "dof_pos": 1.0, "dof_vel": 0.05,
        },
    }
    
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_upward_velocity": 1.2,
        "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0, "base_height": -50.0,
            "action_rate": -0.005, "similar_to_default": -0.1,
        }
    }
    
    command_cfg = {
        "num_commands": 5,
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.6, 0.6],
        "height_range": [0.2, 0.4],
        "jump_range": [0.5, 1.5],
    }
    
    # Create environment
    env = Go2Env(
        num_envs=1, env_cfg=env_cfg, obs_cfg=obs_cfg,
        reward_cfg=reward_cfg, command_cfg=command_cfg,
        show_viewer=True, device="cpu"
    )
    
    print("\n" + "="*60)
    print("Walking Demo - Robot will attempt to walk forward")
    print("="*60)
    print("\nNOTE: Without a trained policy, the robot may fall.")
    print("To train a policy, run: python src/go2_train.py")
    print("\nPress Ctrl+C to exit.")
    print("="*60 + "\n")
    
    # Reset environment
    obs, _ = env.reset()
    
    try:
        for i in range(2000):
            # Set walking command: forward at 0.5 m/s
            if i < 100:
                # Start slow
                env.commands[:] = torch.tensor([[0.3, 0.0, 0.0, 0.3, 0.0]], device="cpu")
            elif i < 1000:
                # Walk forward
                env.commands[:] = torch.tensor([[0.8, 0.0, 0.0, 0.3, 0.0]], device="cpu")
            elif i < 1500:
                # Turn right while walking
                env.commands[:] = torch.tensor([[0.5, 0.0, -0.3, 0.3, 0.0]], device="cpu")
            else:
                # Walk backward
                env.commands[:] = torch.tensor([[-0.3, 0.0, 0.0, 0.3, 0.0]], device="cpu")
            
            # Zero actions - robot will try to track commands using default controller
            actions = torch.zeros((1, 12), device="cpu")
            obs, privileged_obs, rewards, dones, extras = env.step(actions)
            
            if i % 100 == 0:
                cmd = env.commands[0]
                print(f"Step {i}: Cmd=[vx:{cmd[0]:.1f}, vy:{cmd[1]:.1f}, vyaw:{cmd[2]:.1f}]")
    
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    
    print("Demo completed!")

if __name__ == "__main__":
    main()
