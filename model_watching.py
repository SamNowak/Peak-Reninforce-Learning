import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO, SAC
import peak_env  # Ensure peak_env.py is in path


def main(args):
    # Create environment and wrap for recording
    env = gym.make("Peak-v4")
    env = RecordVideo(env, video_folder=args.video_folder, episode_trigger=lambda x: True)

    # Load model
    if args.algo.lower() == 'ppo':
        model = PPO.load(args.model_path)
    elif args.algo.lower() == 'sac':
        model = SAC.load(args.model_path)
    else:
        raise ValueError("Algorithm must be 'ppo' or 'sac'")

    # Playback
    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)
        print(f"Episode {ep + 1} recorded to {args.video_folder}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record trained Peak agent gameplay")
    parser.add_argument("--algo", type=str, required=True, choices=['ppo', 'sac'], help="Algorithm of model to load")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model .zip file")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--video_folder", type=str, default="videos", help="Folder to save recorded videos")
    args = parser.parse_args()
    main(args)
