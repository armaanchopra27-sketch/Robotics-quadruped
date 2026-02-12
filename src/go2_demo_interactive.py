"""
Interactive demo to control the Go2 quadruped robot in Genesis simulator.
Use keyboard to control the robot's movement.

Controls:
    W/S: Forward/Backward velocity
    A/D: Left/Right velocity
    Q/E: Turn left/right (angular velocity)
    R: Reset velocities to zero
    ESC: Exit
"""

import torch
import genesis as gs
from go2_env import Go2Env
from pynput import keyboard
import threading
import time

class RobotController:
    def __init__(self):
        self.lin_vel_x = 0.0  # Forward/backward
        self.lin_vel_y = 0.0  # Left/right
        self.ang_vel_z = 0.0  # Rotation
        self.height = 0.3     # Target height
        self.running = True
        
        # Velocity increments
        self.vel_increment = 0.1
        self.ang_increment = 0.1
        self.max_lin_vel = 1.0
        self.max_ang_vel = 0.6
        
    def on_press(self, key):
        try:
            if key.char == 'w':
                self.lin_vel_x = min(self.lin_vel_x + self.vel_increment, self.max_lin_vel)
                print(f"Forward velocity: {self.lin_vel_x:.2f}")
            elif key.char == 's':
                self.lin_vel_x = max(self.lin_vel_x - self.vel_increment, -self.max_lin_vel)
                print(f"Forward velocity: {self.lin_vel_x:.2f}")
            elif key.char == 'a':
                self.lin_vel_y = min(self.lin_vel_y + self.vel_increment, self.max_lin_vel)
                print(f"Sideways velocity: {self.lin_vel_y:.2f}")
            elif key.char == 'd':
                self.lin_vel_y = max(self.lin_vel_y - self.vel_increment, -self.max_lin_vel)
                print(f"Sideways velocity: {self.lin_vel_y:.2f}")
            elif key.char == 'q':
                self.ang_vel_z = min(self.ang_vel_z + self.ang_increment, self.max_ang_vel)
                print(f"Angular velocity: {self.ang_vel_z:.2f}")
            elif key.char == 'e':
                self.ang_vel_z = max(self.ang_vel_z - self.ang_increment, -self.max_ang_vel)
                print(f"Angular velocity: {self.ang_vel_z:.2f}")
            elif key.char == 'r':
                self.lin_vel_x = 0.0
                self.lin_vel_y = 0.0
                self.ang_vel_z = 0.0
                print("Velocities reset to zero")
        except AttributeError:
            if key == keyboard.Key.esc:
                print("ESC pressed - exiting...")
                self.running = False
                return False
                
    def get_commands(self):
        return torch.tensor([[self.lin_vel_x, self.lin_vel_y, self.ang_vel_z, self.height, 0.0]], 
                          dtype=torch.float32, device='cpu')

def main():
    # Initialize Genesis
    gs.init(backend=gs.constants.backend.cpu, logging_level="warning")
    
    print("Creating Go2 quadruped environment...")
    
    # Configuration for the environment
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
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
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
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "jump_upward_velocity": 1.2,
        "jump_reward_steps": 50,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
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
    
    # Create environment with viewer enabled
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
        device="cpu"
    )
    
    # Create controller
    controller = RobotController()
    
    # Start keyboard listener in separate thread
    listener = keyboard.Listener(on_press=controller.on_press)
    listener.start()
    
    print("\n" + "="*60)
    print("Interactive Demo Started!")
    print("="*60)
    print("\nKeyboard Controls:")
    print("  W - Increase forward velocity")
    print("  S - Decrease forward velocity (or go backward)")
    print("  A - Move left")
    print("  D - Move right")
    print("  Q - Turn left (counter-clockwise)")
    print("  E - Turn right (clockwise)")
    print("  R - Reset all velocities to zero")
    print("  ESC - Exit")
    print("\nA visualization window should be open.")
    print("The robot will respond to your keyboard commands.")
    print("="*60 + "\n")
    
    # Reset environment
    obs, _ = env.reset()
    
    # Override commands to manual control
    step_count = 0
    
    try:
        while controller.running:
            # Get manual commands from controller
            env.commands[:] = controller.get_commands()
            
            # Use trained policy or simple PD controller
            # For now, use zero actions (the environment will use PD control)
            actions = torch.zeros((1, 12), device="cpu")
            
            obs, privileged_obs, rewards, dones, extras = env.step(actions)
            
            step_count += 1
            
            # Display status every 2 seconds (50 steps at 25Hz)
            if step_count % 50 == 0:
                print(f"[Step {step_count}] Cmd: x={controller.lin_vel_x:.2f} y={controller.lin_vel_y:.2f} yaw={controller.ang_vel_z:.2f}")
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    listener.stop()
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
