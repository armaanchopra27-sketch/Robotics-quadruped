import random
import torch
import math
import genesis as gs  # type: ignore
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat  # type: ignore


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower

def gs_rand_gaussian(mean, min, max, n_std, shape, device):
    mean_tensor = mean.expand(shape).to(device)
    std_tensor = torch.full(shape, (max - min)/ 4.0 * n_std, device=device)
    return torch.clamp(torch.normal(mean_tensor, std_tensor), min, max)

def gs_additive(base, increment):
    return base + increment



class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device="cuda", add_camera = False):
        self.device = torch.device(device)

        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(  # type: ignore
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),  # type: ignore
            viewer_options=gs.options.ViewerOptions(  # type: ignore
                max_FPS=int(0.5 / self.dt),
                camera_pos=(3.5, 0.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=num_envs, show_world_frame=False),  # type: ignore
            rigid_options=gs.options.RigidOptions(  # type: ignore
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,  # type: ignore
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))  # type: ignore

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(  # type: ignore
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )
        
        # self.cam_0 : gs.Camera = None
        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs, env_spacing=(1.0, 1.0))

        # names to indices
        self.motor_dofs = [self.robot.get_joint(name).dof_idx_local for name in self.env_cfg["dof_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)  # type: ignore

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=gs.tc_float).repeat(  # type: ignore
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)  # type: ignore
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)  # type: ignore
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"], self.obs_scales["lin_vel"], self.obs_scales["lin_vel"]] ,
            device=self.device,
            dtype=gs.tc_float,  # type: ignore
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)  # type: ignore
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["dof_names"]],
            device=self.device,
            dtype=gs.tc_float,  # type: ignore
        )
        
        self.jump_toggled_buf = torch.zeros((self.num_envs,), device=self.device)
        self.jump_target_height = torch.zeros((self.num_envs,), device=self.device)
        
        self.extras = dict()  # extra information for logging

    def _resample_commands(self, envs_idx):
        # self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)
        # self.commands[envs_idx, 0] =  gs_additive(self.last_actions[envs_idx, 0], self.command_cfg["lin_vel_x_range"][0] + (self.command_cfg["lin_vel_x_range"][1] - self.command_cfg["lin_vel_x_range"][0]) * torch.sin(2 * math.pi * self.episode_length_buf[envs_idx] / 300))
        self.commands[envs_idx, 0] =  gs_rand_gaussian(self.last_actions[envs_idx, 0], *self.command_cfg["lin_vel_x_range"],  2.0, (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 1] =  gs_rand_gaussian(self.last_actions[envs_idx, 1], *self.command_cfg["lin_vel_y_range"],  2.0, (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 2] =  gs_rand_gaussian(self.last_actions[envs_idx, 2], *self.command_cfg["ang_vel_range"],  2.0, (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 3] =  gs_rand_gaussian(self.last_actions[envs_idx, 3], *self.command_cfg["height_range"],  0.5,(len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 4] = 0.0
        
        # scale lin_vel and ang_vel proportionally to the height difference between the target and default height
        height_diff_scale = 0.5 + abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"])/ (self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]) * 0.5
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale
        
    def _sample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 3] = gs_rand_float(*self.command_cfg["height_range"], (len(envs_idx),), self.device)  # type: ignore
        self.commands[envs_idx, 4] = 0.0
        
        # scale lin_vel and ang_vel proportionally to the height difference between the target and default height
        height_diff_scale = 0.5 + abs(self.commands[envs_idx, 3] - self.reward_cfg["base_height_target"])/ (self.command_cfg["height_range"][1] - self.reward_cfg["base_height_target"]) * 0.5
        self.commands[envs_idx, 0] *= height_diff_scale
        self.commands[envs_idx, 1] *= height_diff_scale
        self.commands[envs_idx, 2] *= height_diff_scale
    
    def _sample_jump_commands(self, envs_idx):
        self.commands[envs_idx, 4] = gs_rand_float(*self.command_cfg["jump_range"], (len(envs_idx),), self.device)  # type: ignore
        
    def step(self, actions, is_train=True):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat)
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motor_dofs)

        # resample commands, it is a variable that holds the indices of environments that need to be resampled or reset. 
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        if is_train:
            # self._resample_commands(all_envs_idx)
            self._sample_commands(envs_idx)
            # Idxs with probability of 5% to sample random commands
            ranomd_idxs_1 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_commands(ranomd_idxs_1)
            
            random_idxs_2 = torch.randperm(self.num_envs)[:int(self.num_envs * 0.05)]
            self._sample_jump_commands(random_idxs_2)
            
        # Update jump_toggled_buf if command 4 goes from 0 -> non-zero
        jump_cmd_now = (self.commands[:, 4] > 0.0).float()
        toggle_mask = ((self.jump_toggled_buf == 0.0) & (jump_cmd_now > 0.0)).float()
        self.jump_toggled_buf += toggle_mask * self.reward_cfg["jump_reward_steps"]  # stay 'active' for n steps, for example
        self.jump_toggled_buf = torch.clamp(self.jump_toggled_buf - 1.0, min=0.0)
        # Update jump_target_height if command 4 goes from 0 -> non-zero
        self.jump_target_height = torch.where(jump_cmd_now > 0.0, self.commands[:, 4], self.jump_target_height)
        
        # print(f'jump_toggled_buf: {self.jump_toggled_buf}, jump_target_height: {self.jump_target_height}, commands: {self.commands}')
        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)  # type: ignore
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())    
        
        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 5
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                (self.jump_toggled_buf / self.reward_cfg["jump_reward_steps"]).unsqueeze(-1),  # 1
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        
        # Reset jump command
        self.commands[:, 4] = 0.0

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True
        self.jump_toggled_buf[envs_idx] = 0.0
        self.jump_target_height[envs_idx] = 0.0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._sample_commands(envs_idx)
        
        # set target height command to default height
        self.commands[envs_idx, 3] = self.reward_cfg["base_height_target"]
        

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_forward_motion(self):
        # +1.5 × v_x (forward velocity)
        return self.base_lin_vel[:, 0]  # x-axis velocity

    def _reward_height_consistency(self):
        # +2.0 × exp(-5(z-0.3)²)
        z = self.base_pos[:, 2]
        target = self.reward_cfg["base_height_target"]
        return torch.exp(-5.0 * (z - target) ** 2)

    def _reward_sine_gait(self):
        # +3.0 × Σexp(-(pos-sine)²)
        # Simple sine wave for now - can be improved later
        # Oscillate around default position with time
        time = self.episode_length_buf.float() * self.dt
        frequency = 1.0  # 1 Hz stride
        
        # Create sine wave target for each joint [num_envs, num_actions]
        sine_targets = self.default_dof_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        for i in range(self.num_actions):
            amplitude = 0.3  # radians
            phase = (i % 4) * 3.14159 / 2  # Different phase per leg
            sine_targets[:, i] += amplitude * torch.sin(2 * 3.14159 * frequency * time + phase)
        
        # Sum of exponential errors
        errors = torch.exp(-torch.square(self.dof_pos - sine_targets))
        return torch.sum(errors, dim=1) / self.num_actions  # Average over joints

    def _reward_upright_posture(self):
        # +1.0 × cos(pitch)×cos(roll)
        # projected_gravity is [gx, gy, gz] in body frame
        pitch = torch.atan2(self.projected_gravity[:, 0], self.projected_gravity[:, 2])
        roll = torch.atan2(self.projected_gravity[:, 1], self.projected_gravity[:, 2])
        return torch.cos(pitch) * torch.cos(roll)

    def _reward_energy_smoothness(self):
        # -0.1 × Σ(Δaction)² (penalize, so return positive for small changes)
        # Will be multiplied by -0.1 in reward_scales
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_death_penalty(self):
        # -20.0 on termination (this will be applied when robot falls)
        # Return 1.0 when terminated, 0.0 otherwise
        # The -20.0 scale will be applied via reward_scales
        return self.reset_buf.float() 
