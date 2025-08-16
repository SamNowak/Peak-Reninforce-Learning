import sys
import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import numpy as np
import peak_env  # Ensure peak_env.py is in the path
from peak_env_wrapper import make_wrapped_env  # Import the wrapper


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
        
        # Attention mechanism for focusing on important regions
        self.attention = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
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
        
        # Final linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
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


class CurriculumCallback(CheckpointCallback):
    """Custom callback for curriculum learning and monitoring"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_difficulty = 'easy'
        self.best_reward = -float('inf')
        
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
            
            # Adjust difficulty every 100 episodes
            if len(self.episode_rewards) % 100 == 0 and len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                
                # Simple curriculum logic
                if avg_reward > 50 and self.current_difficulty == 'easy':
                    self.current_difficulty = 'medium'
                    print(f"Advancing to medium difficulty! Avg reward: {avg_reward:.2f}")
                elif avg_reward > 100 and self.current_difficulty == 'medium':
                    self.current_difficulty = 'hard'
                    print(f"Advancing to hard difficulty! Avg reward: {avg_reward:.2f}")
                elif avg_reward < 20 and self.current_difficulty != 'easy':
                    self.current_difficulty = 'easy'
                    print(f"Returning to easy difficulty. Avg reward: {avg_reward:.2f}")
                
                # Log statistics
                print(f"Episode {len(self.episode_rewards)}: Avg Reward = {avg_reward:.2f}, "
                      f"Avg Length = {np.mean(self.episode_lengths[-100:]):.0f}")
        
        return super()._on_step()



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

    # Create vectorized environment WITH WRAPPER
    print(f"Creating {args.n_envs} parallel environments...")

    # Import the wrapper function
    from peak_env_wrapper import make_wrapped_env

    if args.vec_env_type == "dummy":
        env_fns = [make_wrapped_env(i, seed=args.seed, log_dir=args.log_dir,
                                    difficulty=args.start_difficulty, frame_skip=args.frame_skip,
                                    use_simplified=args.use_simplified)
                   for i in range(args.n_envs)]
        vec_env = DummyVecEnv(env_fns)
    else:
        env_fns = [make_wrapped_env(i, seed=args.seed, log_dir=args.log_dir,
                                    difficulty=args.start_difficulty, frame_skip=args.frame_skip,
                                    use_simplified=args.use_simplified)
                   for i in range(args.n_envs)]
        vec_env = SubprocVecEnv(env_fns, start_method='spawn')

    vec_env = VecMonitor(vec_env)
    
    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    checkpoint_cb = CurriculumCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_peak",
        save_vecnormalize=True,
        save_replay_buffer=False,
        verbose=1
    )
    
    # Evaluation environment
    from peak_env_wrapper import FlattenActionWrapper, SimplifiedPeakWrapper

    # Create eval env with wrapper
    eval_env = gym.make("Peak-v4", obs_mode='pixels', difficulty='medium', frame_skip=args.frame_skip)
    if args.use_simplified:
        eval_env = SimplifiedPeakWrapper(eval_env)
    else:
        eval_env = FlattenActionWrapper(eval_env)
    eval_env = Monitor(
        eval_env,
        filename=os.path.join(args.log_dir, "ppo_eval.csv"),
        info_keywords=('height', 'stamina', 'success_rate')
    )
    
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("best_model", "ppo"),
        log_path=os.path.join(args.log_dir, "ppo_eval"),
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
            vf=[256, 256]   # Critic network
        ),
        activation_fn=nn.ReLU,
        ortho_init=True,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(eps=1e-5)
    )
    
    # Initialize or load model
    if args.resume:
        print(f"Resuming PPO training from {args.resume}")
        model = PPO.load(
            args.resume, 
            env=vec_env,
            device=device,
            custom_objects={'learning_rate': linear_schedule(args.learning_rate)}
        )
        reset_timesteps = False
    else:
        print("Starting new PPO training")
        
        # Determine learning rate schedule
        if args.use_linear_schedule:
            learning_rate = linear_schedule(args.learning_rate)
        else:
            learning_rate = args.learning_rate
        
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            clip_range_vf=None,  # No value function clipping
            normalize_advantage=True,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            use_sde=args.use_sde,  # State-dependent exploration
            sde_sample_freq=args.sde_sample_freq,
            target_kl=0.01,  # Early stopping for KL divergence
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            device=device,
            seed=args.seed
        )
        reset_timesteps = True
    
    # Train the agent
    print(f"\nTraining for {args.total_timesteps} timesteps...")
    print(f"Using {args.n_envs} parallel environments")
    print(f"Batch size: {args.batch_size}, Buffer size: {args.n_steps * args.n_envs}")
    print(f"Updates will occur every {args.n_steps * args.n_envs} steps")
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
            tb_log_name="ppo_run"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    model.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")
    
    # Save training statistics
    import json
    stats = {
        'episode_rewards': checkpoint_cb.episode_rewards,
        'episode_lengths': checkpoint_cb.episode_lengths,
        'final_difficulty': checkpoint_cb.current_difficulty,
        'best_reward': checkpoint_cb.best_reward,
        'total_episodes': len(checkpoint_cb.episode_rewards)
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
    
    vec_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on PeakEnv with advanced features")
    
    # Environment settings
    parser.add_argument("--n_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--vec_env_type", type=str, default="dummy", choices=["dummy", "subproc"],
                       help="Type of vectorized environment")
    parser.add_argument("--use_simplified", action="store_true",
                        help="Use simplified action space (recommended for beginners)")
    parser.add_argument("--start_difficulty", type=str, default="easy", 
                       choices=["easy", "medium", "hard"], help="Starting difficulty")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip for faster processing")
    
    # Training settings
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage even if GPU available")
    
    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--use_linear_schedule", action="store_true", help="Use linear LR schedule")
    parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per environment")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm")
    parser.add_argument("--use_sde", action="store_true", help="Use state-dependent exploration")
    parser.add_argument("--sde_sample_freq", type=int, default=4, help="SDE sample frequency")
    
    # Logging and checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ppo", 
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=50000, 
                       help="Checkpoint save frequency")
    parser.add_argument("--eval_freq", type=int, default=20000, 
                       help="Evaluation frequency")
    parser.add_argument("--log_dir", type=str, default="logs/ppo", 
                       help="Directory for logs")
    parser.add_argument("--tensorboard_log", type=str, default="tensorboard/ppo", 
                       help="Tensorboard log directory")
    parser.add_argument("--save_path", type=str, default="ppo_peak_final", 
                       help="Path to save final model")
    
    args = parser.parse_args()
    main(args)