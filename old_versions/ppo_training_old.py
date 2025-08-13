import sys
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym
import peak_env  # Ensure peak_env.py is in the path

def make_env(rank, seed=0, log_dir="./logs/ppo"):
    def _init():
        env = gym.make("Peak-v4")
        os.makedirs(log_dir, exist_ok=True)
        monitor_path = os.path.join(log_dir, f"monitor_{rank}.csv")
        env = Monitor(env, filename=monitor_path, info_keywords=('map_complexity',))
        env.seed(seed + rank)
        return env
    return _init

def main(args):
    # Create vectorized env with monitoring
    num_envs = 4
    env_fns = [make_env(i, seed=args.seed, log_dir=args.log_dir) for i in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    # Callbacks
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=args.checkpoint_dir,
        name_prefix="ppo_peak"
    )
    eval_env = Monitor(gym.make("Peak-v4"), filename=os.path.join(args.log_dir, "ppo_eval.csv"), info_keywords=('map_complexity',))
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join("best_model", "ppo"),
        log_path=os.path.join(args.log_dir, "ppo_eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False
    )

    # Initialize or load model
    if args.resume:
        model = PPO.load(args.resume, env=vec_env)
        reset_timesteps = False
        print(f"Resuming PPO training from {args.resume}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=args.tensorboard_log
        )
        reset_timesteps = True

    # Train the agent
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=reset_timesteps,
        callback=[checkpoint_cb, eval_cb]
    )

    # Save final model
    model.save(args.save_path)
    print(f"Training complete. Model saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on PeakEnv with resume support")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint zip to resume from")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=200_000, help="Total timesteps for learning")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/ppo", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=50000, help="Checkpoint save frequency (in steps)")
    parser.add_argument("--eval_freq", type=int, default=20000, help="Evaluation frequency (in steps)")
    parser.add_argument("--log_dir", type=str, default="logs/ppo", help="Directory for Monitor logs")
    parser.add_argument("--tensorboard_log", type=str, default="tensorboard/ppo", help="Tensorboard log dir")
    parser.add_argument("--save_path", type=str, default="ppo_peak_final", help="Path to save final model")
    args = parser.parse_args()
    main(args)
