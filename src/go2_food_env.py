"""
Food-seeking environment for hierarchical RL.

The robot must use its camera to locate food and navigate to it.
High-level policy (vision) → Commands → Low-level policy (locomotion)

This mimics the biological brain's visual-motor loop.
"""

import torch
import numpy as np
import genesis as gs
from go2_env import Go2Env


class Go2FoodEnv(Go2Env):
    """
    Extended environment with food objects for vision-based navigation.
    
    The robot:
    1. Sees food through camera (visual cortex)
    2. Decides movement command (prefrontal cortex) 
    3. Executes command with trained locomotion (motor cortex)
    """
    
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        device,
        num_food=3,
        food_spawn_radius=8.0,
        food_collect_radius=1.5,
    ):
        """
        Args:
            num_food: Number of food items per environment
            food_spawn_radius: How far from origin to spawn food (meters)
            food_collect_radius: Distance to collect food (meters)
        """
        self.num_food = num_food
        self.food_spawn_radius = food_spawn_radius
        self.food_collect_radius = food_collect_radius
        
        # Enable camera (to see food) AND lidar (to measure distances/obstacles)
        super().__init__(
            num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
            device=device,
            add_camera=True,
            add_lidar=True
        )
        
        # Food state tracking (positions only, visualized with debug spheres)
        self.food_positions = torch.zeros(
            (num_envs, num_food, 3), device=device, dtype=torch.float32
        )
        self.food_active = torch.ones(
            (num_envs, num_food), device=device, dtype=torch.bool
        )
        self.food_collected_count = torch.zeros(num_envs, device=device, dtype=torch.int32)
        
        # For science fair metrics
        self.total_food_collected = 0
        self.episode_food_history = []
    
    def reset(self):
        """Reset environment and randomize food positions"""
        obs = super().reset()
        
        # Place food randomly
        self._spawn_food(torch.arange(self.num_envs, device=self.device))
        
        return obs
    
    def _spawn_food(self, env_ids):
        """
        Spawn food at random positions around the environment.
        
        Args:
            env_ids: Which environments to spawn food in
        """
        num_envs = len(env_ids)
        
        # Random positions in a circle
        angles = torch.rand((num_envs, self.num_food), device=self.device) * 2 * np.pi
        radii = torch.rand((num_envs, self.num_food), device=self.device) * self.food_spawn_radius
        
        # Convert polar to Cartesian
        self.food_positions[env_ids, :, 0] = radii * torch.cos(angles)  # x
        self.food_positions[env_ids, :, 1] = radii * torch.sin(angles)  # y
        self.food_positions[env_ids, :, 2] = 0.2  # z (slightly above ground)
        
        # Mark all food as active
        self.food_active[env_ids, :] = True
    
    def step(self, actions, is_train=True):
        """Step with food collection checking"""
        obs, privileged_obs, rewards, dones, infos = super().step(actions, is_train)
        
        # Check for food collection
        food_rewards = self._check_food_collection()
        
        # Add food rewards to total
        rewards = rewards + food_rewards
        
        # Respawn food in finished episodes
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            # Track food collected this episode
            for env_id in done_ids:
                count = self.food_collected_count[env_id].item()
                self.episode_food_history.append(count)
                self.total_food_collected += count
            
            # Reset counters and spawn new food
            self.food_collected_count[done_ids] = 0
            self._spawn_food(done_ids)
        
        # Add food stats to info
        if 'episode' in infos and infos['episode'] is not None:
            infos['episode']['food_collected'] = self.food_collected_count.float().mean().item()
            if len(self.episode_food_history) > 0:
                infos['episode']['avg_food_per_episode'] = np.mean(self.episode_food_history[-100:])
        
        return obs, privileged_obs, rewards, dones, infos
    
    def _check_food_collection(self):
        """
        Check if robot is close enough to collect food.
        
        Returns:
            rewards: Food collection rewards [num_envs]
        """
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Robot position [num_envs, 3]
        robot_pos = self.base_pos
        
        for food_idx in range(self.num_food):
            # Food position [num_envs, 3]
            food_pos = self.food_positions[:, food_idx, :]
            
            # Distance to this food [num_envs]
            dist = torch.norm(robot_pos - food_pos, dim=1)
            
            # Collect if close enough AND active
            collected = (dist < self.food_collect_radius) & self.food_active[:, food_idx]
            
            # Reward collection
            rewards[collected] += 100.0  # Sparse reward - big bonus!
            
            # Update counters
            self.food_collected_count[collected] += 1
            
            # Deactivate collected food
            self.food_active[:, food_idx] = self.food_active[:, food_idx] & ~collected
        
        # Dense reward: Approaching nearest active food
        approach_reward = self._compute_approach_reward()
        
        return rewards + approach_reward
    
    def _compute_approach_reward(self):
        """
        Reward for approaching the nearest food.
        This provides dense feedback for learning.
        
        Returns:
            rewards: Approach rewards [num_envs]
        """
        robot_pos = self.base_pos  # [num_envs, 3]
        
        # Find distance to nearest active food
        min_dist = torch.full((self.num_envs,), float('inf'), device=self.device)
        
        for food_idx in range(self.num_food):
            food_pos = self.food_positions[:, food_idx, :]
            dist = torch.norm(robot_pos - food_pos, dim=1)
            
            # Only consider active food
            dist = torch.where(
                self.food_active[:, food_idx],
                dist,
                torch.tensor(float('inf'), device=self.device)
            )
            
            min_dist = torch.min(min_dist, dist)
        
        # Reward inversely proportional to distance
        # Closer = higher reward
        approach_reward = -0.1 * min_dist
        
        # Clip to avoid huge negative rewards if very far
        approach_reward = torch.clamp(approach_reward, min=-10.0, max=0.0)
        
        return approach_reward
    
    def get_camera_obs(self):
        """
        Get camera observations for vision policy.
        
        Returns:
            images: RGB images [num_envs, 3, H, W]
        """
        if not hasattr(self, 'camera') or self.camera is None:  # type: ignore
            # Dummy observation if camera not set up
            return torch.zeros((self.num_envs, 3, 120, 160), device=self.device)
        
        # Get images from Genesis camera
        images = self.camera.get_images()  # type: ignore
        
        # Convert to tensor if needed
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).to(self.device)
        
        # Rearrange: [N, H, W, C] → [N, C, H, W]
        images = images.permute(0, 3, 1, 2).float() / 255.0
        
        return images
    
    def get_nearest_food_direction(self):
        """
        Get direction vector to nearest food (for debugging/analysis).
        
        Returns:
            directions: Unit vectors pointing to nearest food [num_envs, 2]
        """
        robot_pos = self.base_pos[:, :2]  # Only x,y
        
        nearest_food_pos = torch.zeros((self.num_envs, 2), device=self.device)
        min_dist = torch.full((self.num_envs,), float('inf'), device=self.device)
        
        for food_idx in range(self.num_food):
            food_pos = self.food_positions[:, food_idx, :2]
            dist = torch.norm(robot_pos - food_pos, dim=1)
            
            # Only consider active
            active_mask = self.food_active[:, food_idx]
            better_mask = (dist < min_dist) & active_mask
            
            nearest_food_pos[better_mask] = food_pos[better_mask]
            min_dist[better_mask] = dist[better_mask]
        
        # Direction vector
        direction = nearest_food_pos - robot_pos
        
        # Normalize
        direction_norm = torch.norm(direction, dim=1, keepdim=True)
        direction = direction / (direction_norm + 1e-8)
        
        return direction
