import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenActionWrapper(gym.ActionWrapper):
    """Convert Dict action space to flat Box for SB3 compatibility."""
    def __init__(self, env):
        super().__init__(env)
        self.original_action_space = env.action_space
        self.n_keys = len(self.original_action_space['keys'].nvec)
        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(self.n_keys + 3,), dtype=np.float32)
        self.discrete_dims = self.original_action_space['keys'].nvec

    def action(self, action):
        keys_continuous = action[:self.n_keys]
        camera_action = action[self.n_keys:self.n_keys + 2]
        scroll_action = action[self.n_keys + 2:self.n_keys + 3]
        keys_discrete = []
        for cont_val, n_actions in zip(keys_continuous, self.discrete_dims):
            discrete_val = int(np.clip((cont_val + 1.0) * (n_actions - 1) / 2.0 + 0.5, 0, n_actions - 1))
            keys_discrete.append(discrete_val)
        return {
            'keys': np.array(keys_discrete, dtype=np.int64),
            'camera': np.clip(camera_action, -1.0, 1.0).astype(np.float32),
            'scroll': np.clip(scroll_action, -1.0, 1.0).astype(np.float32)
        }

class SimplifiedPeakWrapper(gym.Wrapper):
    """Smaller continuous action space for faster learning/debugging."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(9,), dtype=np.float32)

    def step(self, action):
        h_move = int(np.clip((action[0] + 1.0) * 1.5, 0, 2))
        v_move = int(np.clip((action[1] + 1.0) * 1.5, 0, 2))
        jump = int(action[2] > 0)
        climb = int(action[3] > 0)
        camera = action[4:6]
        dict_action = {
            'keys': np.array([h_move, v_move, jump, 0, 0, 0, 0, 0, 0, climb], dtype=np.int64),
            'camera': camera.astype(np.float32),
            'scroll': np.array([0.0], dtype=np.float32)
        }
        return self.env.step(dict_action)

def make_wrapped_env(rank, seed=0, log_dir="./logs/ppo",
                     difficulty='easy', frame_skip=2, use_simplified=False,
                     window_title=None, auto_snap_window=True):
    def _init():
        from stable_baselines3.common.utils import set_random_seed
        env = gym.make("Peak-v4",
                       obs_mode='pixels',
                       difficulty=difficulty,
                       frame_skip=frame_skip,
                       window_title=window_title,
                       auto_snap_window=auto_snap_window)
        env = (SimplifiedPeakWrapper(env) if use_simplified else FlattenActionWrapper(env))
        env.action_space.seed(seed + rank)
        set_random_seed(seed + rank)
        return env
    return _init
