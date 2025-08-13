import sys
import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
import numpy as np
import peak_env  # Ensure peak_env.py is in the path


class CustomCNN(BaseFeaturesExtractor):
    """Custom CNN with attention mechanism for Peak visual processing"""
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        # Calculate the correct input channels
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space["pixels"].shape[2]
        
        # CNN for main view
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # CNN for HUD
        self.hud_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output dimensions
        with torch.no_grad():
            sample_pixels = torch.zeros(1, *observation_space["pixels"].shape)
            sample_pixels = sample_pixels.permute(0, 3, 1, 2)  # NHWC to NCHW
            cnn_out = self.cnn(sample_pixels)
            
            sample_hud = torch.zeros(1, *observation_space["hud"].shape)
            sample_hud = sample_hud.permute(0, 3, 1, 2)
            hud_out = self.hud_cnn(sample_hud)
            
            n_flatten = cnn_out.shape[1] + hud_out.shape[1] + observation_space["sensors"].shape[0]
        
        # Final linear layer with layer normalization
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations) -> torch.Tensor:
        # Extract components
        pixels = observations["pixels"].float() / 255.0
        hud = observations["hud"].float() / 255.0
        sensors = observations["sensors"].float()
        
        # Permute to NCHW format for Conv2d
        pixels = pixels.permute(0, 3, 1, 2)
        hud = hud.permute(0, 3, 1, 2)
        
        # Process visual inputs
        cnn_features = self.cnn(pixels)
        hud_features = self.hud_cnn(hud)
        
        # Concatenate all features
        combined = torch.cat([cnn_features, hud_features, sensors], dim=1)
        
        return self.linear(combined)


class SACCallback(CheckpointCallback):
    """Custom callback for SAC training with monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = []
        self.episode_lengths = []
        self.actor_losses = []
        self.critic_losses = []
        self.best_reward = -float('inf')
        self.current_difficulty = 'easy'
        
    def _on_step(self) -> bool:
        # Check for episode end
        if len(self.locals.get("dones", [])) > 0 and self.locals["dones"][0]:
            info = self.locals.get("infos", [{}])[0]
            
            # Extract episode statistics
            if "episode" in info:
                ep_reward = info.get("episode", {}).get("r", 0)
                ep_length = info.get("episode", {}).get("l", 0)
                
                self.episode_rewards.append(ep_reward)
                self.episode_lengths.append(ep_length)
                
                # Update best reward
                if ep_reward > self.best_reward:
                    self.best_reward = ep_reward
                    print(f"New best reward: {self.best_reward:.2f}")
                
                # Log to tensorboard
                if self.logger is not None:
                    self.logger.record("rollout/ep_reward", ep_reward)
                    self.logger.record("rollout/ep_length", ep_length)
                    self.logger.record("rollout/best_reward", self.best_reward)
                    
                    # Log buffer statistics
                    if hasattr(self.model, "replay_buffer"):
                        self.logger.record("train/buffer_size", self.model.replay_buffer.size())
            
            # Print progress every 10 episodes
            if len(self.episode_rewards) % 10 == 0 and len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {avg_length:.0f}, "
                      f"Buffer Size = {self.model.replay_buffer.size()}")
            
            # Adjust difficulty based on performance
            if len(self.episode_rewards) % 100 == 0 and len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                
                if avg_reward > 50 and self.current_difficulty == 'easy':
                    self.current_difficulty = 'medium'
                    print(f"Advancing to medium difficulty! Avg reward: {avg_reward:.2f}")
                elif avg_reward > 100 and self.current_difficulty == 'medium':
                    self.current_difficulty = 'hard'
                    print(f"Advancing to hard difficulty! Avg reward: {avg_reward:.2f}")
                elif avg_reward < 20 and self.current_difficulty != 'easy':
                    self.current_difficulty = 'easy'
                    print(f"Returning to easy difficulty. Avg reward: {avg_reward:.2f}")
        
        return super()._on_step()


def make_env(rank, seed=0, log_dir="./logs/sac", difficulty='easy', frame_skip=2):
    """Create a single environment instance"""
    def _init():
        env = gym.make("Peak-v4",
                      obs_mode='pixels',
                      difficulty=difficulty,
                      frame_skip=frame_skip)
        
        os.makedirs(log_dir, exist_ok=True)
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        env = Monitor(env, filename=monitor_path,
                     info_keywords=('height', 'stamina', 'success_rate'))
        env.action_space.seed(seed + rank)
        set_random_seed(seed + rank)
        return env
    
    return _init


def linear_schedule(initial_value: float):
    """Linear learning rate schedule"""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def main(args):
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create vectorized environment
    print(f"Creating {args.n_envs} parallel environments...")
    
    if args.vec_env_type == "dummy":
        env_fns = [make_env(i, seed=args.seed, log_dir=args.log_dir,
                          difficulty=args.start_difficulty, frame_skip=args.frame_skip)
                  for i in range(args.n_envs)]
        vec_env = DummyVecEnv(env_fns)
    else:
        env_fns = [make_env(i, seed=args.seed, log_dir=args.log_dir,
                          difficulty=args.start_difficulty, frame_skip=args.frame_skip)
                  for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    
    vec_env = VecMonitor(vec_env)
    
    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_cb = SACCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="sac_peak",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    # Evaluation environment
    eval_env = Monitor(
        gym.make("Peak-v4", obs_mode='pixels', difficulty='medium', frame_skip=args.frame_skip),
        filename=os.path.join(args.log_dir, "sac_eval.csv"),
        info_keywords=('height', 'stamina', 'success_rate')
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("best_model", "sac"),
        log_path=os.path.join(args.log_dir, "sac_eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Policy kwargs with custom CNN
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[256, 256],  # Actor network
            qf=[256, 256]   # Critic networks
        ),
        activation_fn=nn.ReLU,
        n_critics=2,  # Twin Q-networks
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    # Initialize or load model
    if args.resume:
        print(f"Resuming SAC training from {args.resume}")
        model = SAC.load(
            args.resume,
            env=vec_env,
            device=device,
            custom_objects={'learning_rate': linear_schedule(args.learning_rate) if args.use_linear_schedule else args.learning_rate}
        )
        
        # Load replay buffer if provided
        if args.resume_buffer:
            print(f"Loading replay buffer from {args.resume_buffer}")
            model.load_replay_buffer(args.resume_buffer)
            print(f"Replay buffer loaded with {model.replay_buffer.size()} samples")
        
        reset_timesteps = False
    else:
        print("Starting new SAC training")
        
        # Determine learning rate schedule
        if args.use_linear_schedule:
            learning_rate = linear_schedule(args.learning_rate)
        else:
            learning_rate = args.learning_rate
        
        # Create replay buffer kwargs
        replay_buffer_kwargs = dict(
            handle_timeout_termination=True
        )
        
        model = SAC(
            "MultiInputPolicy",
            vec_env,
            learning_rate=learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=(args.train_freq, "step"),
            gradient_steps=args.gradient_steps,
            action_noise=None,  # SAC has built-in exploration
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            ent_coef=args.ent_coef,
            target_update_interval=args.target_update_interval,
            target_entropy=args.target_entropy,
            use_sde=args.use_sde,
            sde_sample_freq=args.sde_sample_freq,
            use_sde_at_warmup=False,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            device=device,
            seed=args.seed
        )
        reset_timesteps = True
    
    # Train the agent
    print(f"\nTraining for {args.total_timesteps} timesteps...")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Buffer size: {args.buffer_size}, Batch size: {args.batch_size}")
    print(f"Learning starts after {args.learning_starts} steps")
    print(f"Starting difficulty: {args.start_difficulty}")
    print(f"Frame skip: {args.frame_skip}")
    
    if args.use_sde:
        print(f"Using State-Dependent Exploration (SDE)")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            reset_num_timesteps=reset_timesteps,
            callback=CallbackList([checkpoint_cb, eval_cb]),
            progress_bar=True,
            tb_log_name="sac_run"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model and replay buffer
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")
    
    buffer_path = f"{args.save_path}_replay_buffer.pkl"
    model.save_replay_buffer(buffer_path)
    print(f"Replay buffer saved to {buffer_path}")
    
    # Save training statistics
    import json
    stats = {
        'episode_rewards': checkpoint_cb.episode_rewards,
        'episode_lengths': checkpoint_cb.episode_lengths,
        'final_difficulty': checkpoint_cb.current_difficulty,
        'best_reward': checkpoint_cb.best_reward,
        'total_episodes': len(checkpoint_cb.episode_rewards),
        'final_buffer_size': model.replay_buffer.size()
    }
    
    stats_path = f"{args.save_path}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Training statistics saved to {stats_path}")
    
    # Print summary
    if len(checkpoint_cb.episode_rewards) > 0:
        print("\n" + "="*50)
        print("Training Summary")
        print("="*50)
        print(f"Total Episodes: {len(checkpoint_cb.episode_rewards)}")
        print(f"Best Reward: {checkpoint_cb.best_reward:.2f}")
        print(f"Final Avg Reward (last 100 eps): {np.mean(checkpoint_cb.episode_rewards[-100:]):.2f}")
        print(f"Final Avg Length (last 100 eps): {np.mean(checkpoint_cb.episode_lengths[-100:]):.0f}")
        print(f"Final Difficulty: {checkpoint_cb.current_difficulty}")
        print(f"Final Buffer Size: {model.replay_buffer.size()}")
    
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC on PeakEnv with advanced features")
    
    # Environment settings
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--vec_env_type", type=str, default="dummy", choices=["dummy", "subproc"],
                       help="Type of vectorized environment")
    parser.add_argument("--start_difficulty", type=str, default="easy",
                       choices=["easy", "medium", "hard"], help="Starting difficulty")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip for faster processing")
    
    # Training settings
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--resume_buffer", type=str, default=None, help="Path to replay buffer to load")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU available")
    
    # SAC hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--use_linear_schedule", action="store_true", help="Use linear LR schedule")
    parser.add_argument("--buffer_size", type=int, default=1_000_000, help="Replay buffer size")
    parser.add_argument("--learning_starts", type=int, default=10000, help="Steps before training starts")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--train_freq", type=int, default=1, help="Training frequency")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Gradient steps per env step")
    parser.add_argument("--ent_coef", type=str, default="auto", help="Entropy coefficient")
    parser.add_argument("--target_update_interval", type=int, default=1, help="Target network update interval")
    parser.add_argument("--target_entropy", type=str, default="auto", help="Target entropy")
    parser.add_argument("--use_sde", action="store_true", help="Use state-dependent exploration")
    parser.add_argument("--sde_sample_freq", type=int, default=-1, help="SDE sample frequency")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/sac",
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=50000,
                       help="Checkpoint save frequency")
    parser.add_argument("--eval_freq", type=int, default=20000,
                       help="Evaluation frequency")
    parser.add_argument("--log_dir", type=str, default="logs/sac",
                       help="Directory for logs")
    parser.add_argument("--tensorboard_log", type=str, default="tensorboard/sac",
                       help="Tensorboard log directory")
    parser.add_argument("--save_path", type=str, default="sac_peak_final",
                       help="Path to save final model")
    
    args = parser.parse_args()
    main(args)