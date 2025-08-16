import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any


class FlattenActionWrapper(gym.ActionWrapper):
    """
    Wrapper to convert Dict action space to flat Box space for Stable-Baselines3 compatibility.

    Original action space:
    - keys: MultiDiscrete([3, 3, 2, 2, 2, 2, 2, 2, 2, 2])  # 10 discrete actions
    - camera: Box(-1, 1, (2,))  # 2 continuous
    - scroll: Box(-1, 1, (1,))  # 1 continuous

    Flattened to: Box(-1, 1, (13,))  # All continuous
    """

    def __init__(self, env):
        super().__init__(env)

        # Store original action space
        self.original_action_space = env.action_space

        # Get the actual number of key actions
        self.n_keys = len(self.original_action_space['keys'].nvec)

        # Create flattened continuous action space
        # n_keys discrete actions + 2 camera + 1 scroll
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_keys + 3,),  # Dynamic based on actual keys
            dtype=np.float32
        )

        # Store discrete action dimensions for conversion
        self.discrete_dims = self.original_action_space['keys'].nvec

    def action(self, action):
        """Convert flat continuous action to original Dict format."""

        # Split the flat action array
        keys_continuous = action[:self.n_keys]
        camera_action = action[self.n_keys:self.n_keys + 2]
        scroll_action = action[self.n_keys + 2:self.n_keys + 3]

        # Convert continuous values to discrete for keys
        keys_discrete = []
        for i, (cont_val, n_actions) in enumerate(zip(keys_continuous, self.discrete_dims)):
            # Convert from [-1, 1] to discrete action
            discrete_val = int(np.clip(
                (cont_val + 1.0) * (n_actions - 1) / 2.0 + 0.5,
                0,
                n_actions - 1
            ))
            keys_discrete.append(discrete_val)

        # Construct the Dict action
        dict_action = {
            'keys': np.array(keys_discrete, dtype=np.int64),
            'camera': np.clip(camera_action, -1.0, 1.0).astype(np.float32),
            'scroll': np.clip(scroll_action, -1.0, 1.0).astype(np.float32)
        }

        return dict_action


class SimplifiedPeakWrapper(gym.Wrapper):
    """
    Alternative: Simplified action space that only uses essential controls.
    Reduces complexity for faster learning.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Simplified action space: 
        # 0-1: horizontal movement (left/none/right)
        # 2-3: vertical movement (down/none/up)  
        # 4: jump
        # 5: climb
        # 6-7: camera yaw
        # 8: camera pitch
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32
        )
        
    def step(self, action):
        """Convert simplified action to full Dict format."""
        
        # Parse simplified actions
        h_move = int(np.clip((action[0] + 1.0) * 1.5, 0, 2))  # 0, 1, or 2
        v_move = int(np.clip((action[1] + 1.0) * 1.5, 0, 2))  # 0, 1, or 2
        jump = int(action[2] > 0)  # Binary
        climb = int(action[3] > 0)  # Binary
        camera = action[4:6]  # Continuous
        
        # Create full action dict with defaults for unused actions
        dict_action = {
            'keys': np.array([
                h_move,  # horizontal movement
                v_move,  # vertical movement
                jump,    # jump
                0,       # sprint (disabled for simplicity)
                0,       # crouch (disabled)
                0,       # interact (disabled)
                0,       # drop (disabled)
                0,       # emote (disabled)
                0,       # ping (disabled)
                climb    # climb
            ], dtype=np.int64),
            'camera': camera.astype(np.float32),
            'scroll': np.array([0.0], dtype=np.float32)  # No rope control
        }
        
        return self.env.step(dict_action)


def make_wrapped_env(rank, seed=0, log_dir="./logs/ppo", difficulty='easy', frame_skip=2, use_simplified=False):
    """Create environment with proper wrapper for SB3 compatibility."""
    
    def _init():
        import os
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        
        # Create base environment
        env = gym.make("Peak-v4",
                      obs_mode='pixels',
                      difficulty=difficulty,
                      frame_skip=frame_skip)
        
        # Apply wrapper to fix action space
        if use_simplified:
            env = SimplifiedPeakWrapper(env)  # Simpler action space
        else:
            env = FlattenActionWrapper(env)  # Full action space
        
        # Add monitoring
        os.makedirs(log_dir, exist_ok=True)
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        env = Monitor(env, filename=monitor_path,
                     info_keywords=('height', 'stamina', 'success_rate'))
        
        # Set seed
        env.action_space.seed(seed + rank)
        set_random_seed(seed + rank)
        
        return env
    
    return _init