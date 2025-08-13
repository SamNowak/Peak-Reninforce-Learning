#!/usr/bin/env python3
"""
Peak RL Environment Setup and Testing Utility
This script helps set up the environment, capture templates, and test the setup.
"""

import argparse
import sys
import os
import time
import numpy as np
import cv2
import gymnasium as gym
import peak_env
from peak_env import PeakEnv


def test_environment():
    """Test if the environment is working correctly"""
    print("\n" + "="*50)
    print("Testing Peak Environment")
    print("="*50)
    
    try:
        # Test environment creation
        print("\n1. Creating environment...")
        env = gym.make("Peak-v4")
        print("✓ Environment created successfully")
        
        # Test observation space
        print("\n2. Testing observation space...")
        print(f"   Observation space: {env.observation_space}")
        print("✓ Observation space configured")
        
        # Test action space
        print("\n3. Testing action space...")
        print(f"   Action space: {env.action_space}")
        print("✓ Action space configured")
        
        # Test reset
        print("\n4. Testing environment reset...")
        obs = env.reset()
        if env.obs_mode == 'pixels':
            print(f"   Observation keys: {obs.keys()}")
            print(f"   Pixels shape: {obs['pixels'].shape}")
            print(f"   HUD shape: {obs['hud'].shape}")
            print(f"   Sensors shape: {obs['sensors'].shape}")
        else:
            print(f"   Sensors shape: {obs.shape}")
        print("✓ Environment reset successful")
        
        # Test step
        print("\n5. Testing environment step...")
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"   Reward: {reward:.3f}")
        print(f"   Done: {done}")
        print(f"   Info: {info}")
        print("✓ Environment step successful")
        
        env.close()
        print("\n✅ All environment tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_templates():
    """Interactive template capture for the game"""
    print("\n" + "="*50)
    print("Template Capture Utility")
    print("="*50)
    
    print("\n⚠ IMPORTANT: Make sure Peak game is running and visible!")
    print("This utility will help capture template images for:")
    print("  1. Player character")
    print("  2. Summit marker/flag")
    
    response = input("\nIs the Peak game running and ready? (y/n): ")
    if response.lower() != 'y':
        print("Please start the game first, then run this utility again.")
        return False
    
    try:
        from mss import mss
        import keyboard
        
        sct = mss()
        
        print("\n=== Template Capture ===")
        print("Make sure Peak game is running and visible")
        
        print("\n1. Position player character in center of screen")
        print("   Press 'p' when ready...")
        keyboard.wait('p')
        
        # Capture player template - adjust these coordinates as needed
        player_region = {
            'top': 400,
            'left': 600, 
            'width': 80,
            'height': 100
        }
        player_img = np.array(sct.grab(player_region))[:, :, :3]
        cv2.imwrite('player_template.png', cv2.cvtColor(player_img, cv2.COLOR_RGB2BGR))
        print("✓ Player template saved")
        
        print("\n2. Navigate to summit marker/flag")
        print("   Press 's' when summit is visible...")
        keyboard.wait('s')
        
        # Capture summit template
        summit_region = {
            'top': 300,
            'left': 500,
            'width': 200,
            'height': 150
        }
        summit_img = np.array(sct.grab(summit_region))[:, :, :3]
        cv2.imwrite('summit_template.png', cv2.cvtColor(summit_img, cv2.COLOR_RGB2BGR))
        print("✓ Summit template saved")
        
        print("\nTemplates captured successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Template capture failed: {e}")
        print("Make sure 'keyboard' package is installed: pip install keyboard")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n" + "="*50)
    print("Checking Dependencies")
    print("="*50)
    
    dependencies = {
        'numpy': None,
        'opencv-python': None,
        'gymnasium': None,
        'pyautogui': None,
        'mss': None,
        'stable_baselines3': None,
        'torch': None
    }
    
    missing = []
    
    for name in dependencies.keys():
        try:
            if name == 'opencv-python':
                import cv2
            elif name == 'pyautogui':
                import pyautogui
            elif name == 'mss':
                from mss import mss
            elif name == 'stable_baselines3':
                import stable_baselines3
            elif name == 'torch':
                import torch
            elif name == 'numpy':
                import numpy
            elif name == 'gymnasium':
                import gymnasium
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install numpy opencv-python gym pyautogui mss stable-baselines3 torch")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def test_training_setup():
    """Test if training scripts are properly configured"""
    print("\n" + "="*50)
    print("Testing Training Setup")
    print("="*50)
    
    try:
        # Check if training scripts exist
        scripts = ['ppo_training.py', 'sac_training.py', 'model_watching.py']
        
        for script in scripts:
            if os.path.exists(script):
                print(f"✓ {script} found")
            else:
                print(f"✗ {script} NOT found")
        
        # Check for required directories
        print("\nChecking directories...")
        dirs = ['checkpoints', 'logs', 'tensorboard', 'best_model', 'videos']
        
        for dir_name in dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                print(f"✓ Created {dir_name}/")
            else:
                print(f"✓ {dir_name}/ exists")
        
        # Create subdirectories
        subdirs = [
            'checkpoints/ppo', 'checkpoints/sac',
            'logs/ppo', 'logs/sac',
            'tensorboard/ppo', 'tensorboard/sac',
            'best_model/ppo', 'best_model/sac'
        ]
        
        for subdir in subdirs:
            if not os.path.exists(subdir):
                os.makedirs(subdir, exist_ok=True)
                print(f"✓ Created {subdir}/")
        
        print("\n✅ Training setup complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Training setup test failed: {e}")
        return False


def display_info():
    """Display useful information about the setup"""
    print("\n" + "="*50)
    print("Peak RL Training Setup Information")
    print("="*50)
    
    print("\n📁 Project Structure:")
    print("   peak_env.py         - Game environment")
    print("   ppo_training.py     - PPO training script")
    print("   sac_training.py     - SAC training script")
    print("   model_watching.py   - Model evaluation")
    print("   setup_and_test.py   - This utility")
    
    print("\n🎮 Training Commands:")
    print("   Basic PPO training:")
    print("     python ppo_training.py")
    print("\n   Advanced PPO with custom settings:")
    print("     python ppo_training.py --n_envs 8 --total_timesteps 2000000")
    print("\n   SAC training:")
    print("     python sac_training.py")
    print("\n   Resume training:")
    print("     python ppo_training.py --resume checkpoints/ppo/ppo_peak_100000_steps.zip")
    
    print("\n📊 Monitoring:")
    print("   TensorBoard:")
    print("     tensorboard --logdir tensorboard/")
    print("\n   Watch trained model:")
    print("     python model_watching.py --algo ppo --model_path ppo_peak_final.zip")
    
    print("\n💡 Tips:")
    print("   • Start with basic PPO training")
    print("   • Monitor with TensorBoard during training")
    print("   • Save checkpoints frequently")
    print("   • Use multiple environments for faster training")
    print("   • Check template images are captured correctly")


def main():
    parser = argparse.ArgumentParser(
        description="Peak RL Environment Setup and Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_and_test.py --all           # Run all tests
  python setup_and_test.py --test          # Test environment only
  python setup_and_test.py --capture       # Capture template images
  python setup_and_test.py --info          # Show information
        """
    )
    
    parser.add_argument('--all', action='store_true', 
                       help='Run all setup and tests')
    parser.add_argument('--deps', action='store_true',
                       help='Check dependencies')
    parser.add_argument('--test', action='store_true',
                       help='Test environment')
    parser.add_argument('--capture', action='store_true',
                       help='Capture template images')
    parser.add_argument('--training-setup', action='store_true',
                       help='Test training setup')
    parser.add_argument('--info', action='store_true',
                       help='Display setup information')
    
    args = parser.parse_args()
    
    # If no arguments, show info and prompt
    if not any(vars(args).values()):
        display_info()
        print("\n" + "="*50)
        response = input("\nRun all setup and tests? (y/n): ")
        if response.lower() == 'y':
            args.all = True
        else:
            parser.print_help()
            return
    
    results = []
    
    if args.all or args.deps:
        results.append(("Dependencies", check_dependencies()))
    
    if args.all or args.test:
        results.append(("Environment Test", test_environment()))
    
    if args.all or args.capture:
        results.append(("Template Capture", capture_templates()))
    
    if args.all or args.training_setup:
        results.append(("Training Setup", test_training_setup()))
    
    if args.info:
        display_info()
    
    # Summary
    if results:
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        for name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"{name}: {status}")
        
        all_passed = all(r[1] for r in results)
        if all_passed:
            print("\n🎉 All tests passed! You're ready to train!")
            print("\nNext steps:")
            print("1. Make sure Peak game is running")
            print("2. Run: python ppo_training.py")
            print("3. Monitor with: tensorboard --logdir tensorboard/")
        else:
            print("\n⚠ Some tests failed. Please fix the issues above.")


if __name__ == "__main__":
    main()