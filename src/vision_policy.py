"""
Vision-based policy network for food seeking.

This is the "visual cortex + decision making" part of the brain.
Takes camera input → outputs discrete commands for the motor system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionPolicy(nn.Module):
    """
    CNN-based policy that processes camera images to select movement commands.
    
    Architecture mimics visual processing pathway:
    - Early layers: Edge detection (V1)
    - Middle layers: Feature extraction (V2-V4)
    - Late layers: Object recognition (IT cortex)
    - Output: Command selection (Prefrontal cortex)
    """
    
    def __init__(
        self,
        num_commands=6,
        image_size=(120, 160),  # Reduced resolution for 6GB VRAM
        hidden_dim=256,
    ):
        super().__init__()
        
        self.num_commands = num_commands
        
        # Convolutional layers (Visual cortex V1-V4)
        # Input: [batch, 3, 120, 160]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # → [32, 29, 39]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # → [64, 13, 18]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2)  # → [128, 6, 8]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2)  # → [256, 2, 3]
        
        # Calculate flattened size
        self.conv_output_size = 256 * 2 * 3  # 1536
        
        # Fully connected layers (IT cortex → Prefrontal cortex)
        self.fc1 = nn.Linear(self.conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_command = nn.Linear(hidden_dim // 2, num_commands)
        
        # Value head for actor-critic
        self.fc_value = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, image):
        """
        Forward pass through vision network.
        
        Args:
            image: Camera RGB image [batch, 3, H, W]
            
        Returns:
            command_logits: Logits for each discrete command [batch, num_commands]
            value: State value estimate [batch, 1]
        """
        # Visual processing (convolutional layers)
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Decision making (fully connected layers)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Output: Command selection + value estimate
        command_logits = self.fc_command(x)
        value = self.fc_value(x)
        
        return command_logits, value
    
    def get_action(self, image, deterministic=False):
        """
        Sample an action (command) from the policy.
        
        Args:
            image: Camera image [batch, 3, H, W]
            deterministic: If True, select argmax. If False, sample from distribution.
            
        Returns:
            action: Selected command index [batch]
            log_prob: Log probability of selected action [batch]
            value: State value estimate [batch]
        """
        command_logits, value = self.forward(image)
        
        # Create categorical distribution over commands
        dist = torch.distributions.Categorical(logits=command_logits)
        
        if deterministic:
            action = command_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, image, actions):
        """
        Evaluate actions for PPO update.
        
        Args:
            image: Camera images [batch, 3, H, W]
            actions: Actions taken [batch]
            
        Returns:
            log_probs: Log probabilities of actions [batch]
            values: State values [batch]
            entropy: Policy entropy [batch]
        """
        command_logits, values = self.forward(image)
        
        dist = torch.distributions.Categorical(logits=command_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class VisionPolicySmall(nn.Module):
    """
    Smaller/faster vision policy for quick experimentation.
    Uses fewer layers and smaller hidden dimensions.
    """
    
    def __init__(self, num_commands=6):
        super().__init__()
        
        self.num_commands = num_commands
        
        # Simpler conv layers (for 160x120 input)
        # Input: [batch, 3, 120, 160]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)  # → [16, 29, 39]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # → [32, 13, 18]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)  # → [64, 6, 8]
        
        self.conv_output_size = 64 * 6 * 8  # 3072
        
        self.fc1 = nn.Linear(self.conv_output_size, 128)
        self.fc_command = nn.Linear(128, num_commands)
        self.fc_value = nn.Linear(128, 1)
        
    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        
        command_logits = self.fc_command(x)
        value = self.fc_value(x)
        
        return command_logits, value
    
    def get_action(self, image, deterministic=False):
        command_logits, value = self.forward(image)
        dist = torch.distributions.Categorical(logits=command_logits)
        
        if deterministic:
            action = command_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, image, actions):
        command_logits, values = self.forward(image)
        dist = torch.distributions.Categorical(logits=command_logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy


class DualCommandVisionPolicy(nn.Module):
    """
    Dual-command vision policy: outputs two commands simultaneously + degrees.
    
    Command 1 (Movement): Forward/Backward/Stop
    Command 2 (Rotation): Turn Left/Turn Right/Strafe Left/Strafe Right/Stop
    Degrees: 0-90° for turning commands
    
    Example: "Forward + Turn Right 42°"
    """
    
    def __init__(self, image_size=(120, 160)):
        super().__init__()
        
        # Movement commands
        self.movement_commands = ["Forward", "Backward", "Stop"]
        self.num_movement_cmds = 3
        
        # Rotation commands  
        self.rotation_commands = ["Turn Left", "Turn Right", "Strafe Left", "Strafe Right", "Stop"]
        self.num_rotation_cmds = 5
        
        # Convolutional layers (for 160x120 input)
        # Input: [batch, 3, 120, 160]
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4)  # → [16, 29, 39]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # → [32, 13, 18]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)  # → [64, 6, 8]
        
        self.conv_output_size = 64 * 6 * 8  # 3072
        
        # Shared feature extraction
        self.fc_shared = nn.Linear(self.conv_output_size, 256)
        
        # Movement command head
        self.fc_movement = nn.Linear(256, self.num_movement_cmds)
        
        # Rotation command head
        self.fc_rotation = nn.Linear(256, self.num_rotation_cmds)
        
        # Degrees head (continuous, 0-90°)
        self.fc_degrees = nn.Linear(256, 1)
        
        # Value head
        self.fc_value = nn.Linear(256, 1)
        
    def forward(self, image):
        """
        Forward pass.
        
        Args:
            image: Camera RGB image [batch, 3, H, W]
            
        Returns:
            movement_logits: [batch, 3] - Forward/Backward/Stop
            rotation_logits: [batch, 5] - Turn L/R, Strafe L/R, Stop
            degrees: [batch, 1] - Angle 0-90° (sigmoid * 90)
            value: [batch, 1] - State value
        """
        # Conv layers
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Shared features
        x = F.relu(self.fc_shared(x))
        
        # Output heads
        movement_logits = self.fc_movement(x)
        rotation_logits = self.fc_rotation(x)
        degrees = torch.sigmoid(self.fc_degrees(x)) * 90.0  # Scale to 0-90°
        value = self.fc_value(x)
        
        return movement_logits, rotation_logits, degrees, value
    
    def get_action(self, image, deterministic=False):
        """
        Sample dual actions.
        
        Returns:
            movement_action: [batch] - Movement command index
            rotation_action: [batch] - Rotation command index
            degrees: [batch] - Turning angle
            log_prob_movement: [batch]
            log_prob_rotation: [batch]
            value: [batch]
        """
        movement_logits, rotation_logits, degrees, value = self.forward(image)
        
        # Sample from distributions
        movement_dist = torch.distributions.Categorical(logits=movement_logits)
        rotation_dist = torch.distributions.Categorical(logits=rotation_logits)
        
        if deterministic:
            movement_action = movement_logits.argmax(dim=-1)
            rotation_action = rotation_logits.argmax(dim=-1)
        else:
            movement_action = movement_dist.sample()
            rotation_action = rotation_dist.sample()
        
        log_prob_movement = movement_dist.log_prob(movement_action)
        log_prob_rotation = rotation_dist.log_prob(rotation_action)
        
        return (
            movement_action,
            rotation_action,
            degrees.squeeze(-1),
            log_prob_movement,
            log_prob_rotation,
            value.squeeze(-1),
        )
    
    def evaluate_actions(self, image, movement_actions, rotation_actions):
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob_movement: [batch]
            log_prob_rotation: [batch]
            degrees: [batch]
            values: [batch]
            entropy_movement: [batch]
            entropy_rotation: [batch]
        """
        movement_logits, rotation_logits, degrees, values = self.forward(image)
        
        movement_dist = torch.distributions.Categorical(logits=movement_logits)
        rotation_dist = torch.distributions.Categorical(logits=rotation_logits)
        
        log_prob_movement = movement_dist.log_prob(movement_actions)
        log_prob_rotation = rotation_dist.log_prob(rotation_actions)
        
        entropy_movement = movement_dist.entropy()
        entropy_rotation = rotation_dist.entropy()
        
        return (
            log_prob_movement,
            log_prob_rotation,
            degrees.squeeze(-1),
            values.squeeze(-1),
            entropy_movement,
            entropy_rotation,
        )
