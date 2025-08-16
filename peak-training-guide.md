# 🎮 Peak RL Training Guide - Complete Step-by-Step Instructions

## 📋 Pre-Training Setup (One-Time Only)

### Step 1: Environment Setup
```bash
# Create and activate the environment
conda create -n peak-rl python=3.9 cudatoolkit=11.8 -y
conda activate peak-rl

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python gymnasium pyautogui mss stable-baselines3 tensorboard keyboard

# Verify GPU (optional but recommended)
python check_cuda.py
```

### Step 2: Test Everything Works
```bash
# Run complete system check
python setup_and_test.py --all

# If any tests fail, run individual checks:
python setup_and_test.py --deps          # Check dependencies
python setup_and_test.py --training-setup # Create directories
```

### Step 3: Capture Game Templates
```bash
# Start Peak game in windowed mode (1280x720 recommended)
# Then run:
python setup_and_test.py --capture

# Follow prompts:
# 1. Press 'p' with character visible
# 2. Climb to first summit (or find "SUMMIT" text image)
# 3. Press 's' when "SUMMIT" text is visible
```

---

## 🚀 Training Commands - From Beginner to Advanced

### 🟢 BEGINNER - First Training Run
Start with the simplest configuration to ensure everything works:

```bash
# Basic PPO training (safe, stable, slow)
python ppo_training.py
or
python ppo_training.py --n_envs 4 --frame_skip 2

# What this does:
# - Uses 4 parallel environments
# - Trains for 1 million steps
# - Easy difficulty to start
# - Default hyperparameters
# - Saves checkpoints every 50k steps
```

### 🟡 INTERMEDIATE - Optimized Training

#### PPO with Better Hyperparameters
```bash
# Faster training with exploration bonus
python ppo_training.py \
    --n_envs 6 \
    --total_timesteps 2000000 \
    --learning_rate 0.0003 \
    --batch_size 128 \
    --n_steps 2048 \
    --use_sde \
    --ent_coef 0.02 \
    --frame_skip 3

# Explanation of each parameter:
# --n_envs 6              : Run 6 games in parallel (faster learning)
# --total_timesteps 2M    : Train for 2 million steps total
# --learning_rate 0.0003  : Standard learning rate for PPO
# --batch_size 128        : Larger batches for stable learning
# --n_steps 2048          : Steps before each update
# --use_sde               : State-dependent exploration (better exploration)
# --ent_coef 0.02         : Higher entropy for more exploration
# --frame_skip 3          : Process every 3rd frame (3x faster)
```

#### SAC Alternative (Good for Continuous Control)
```bash
# SAC with tuned parameters
python sac_training.py \
    --n_envs 4 \
    --total_timesteps 1500000 \
    --buffer_size 500000 \
    --learning_starts 5000 \
    --batch_size 256 \
    --tau 0.01 \
    --gamma 0.99 \
    --train_freq 4 \
    --gradient_steps 2 \
    --frame_skip 2

# Explanation:
# --buffer_size 500000    : Smaller buffer (uses less RAM)
# --learning_starts 5000  : Start training after 5k steps of exploration
# --tau 0.01              : Faster target network updates
# --train_freq 4          : Update every 4 steps
# --gradient_steps 2      : 2 gradient updates per step
```

### 🔴 ADVANCED - Maximum Performance

#### Aggressive PPO for Fast Learning
```bash
# High-performance configuration (requires good GPU)
python ppo_training.py \
    --n_envs 12 \
    --total_timesteps 5000000 \
    --learning_rate 0.0005 \
    --use_linear_schedule \
    --batch_size 256 \
    --n_steps 1024 \
    --n_epochs 20 \
    --clip_range 0.3 \
    --use_sde \
    --sde_sample_freq 8 \
    --ent_coef 0.01 \
    --vf_coef 0.5 \
    --max_grad_norm 0.7 \
    --frame_skip 4 \
    --start_difficulty easy \
    --checkpoint_freq 25000 \
    --eval_freq 10000

# What makes this "aggressive":
# - 12 parallel environments (high CPU/GPU usage)
# - Higher learning rate with decay
# - More training epochs (20 vs 10)
# - Larger clip range for bigger updates
# - Frequent checkpointing and evaluation
# - Frame skip 4 for maximum speed
```

#### Competition-Ready SAC
```bash
# Maximum sample efficiency SAC
python sac_training.py \
    --n_envs 8 \
    --total_timesteps 3000000 \
    --buffer_size 2000000 \
    --learning_starts 10000 \
    --batch_size 512 \
    --learning_rate 0.0007 \
    --use_linear_schedule \
    --tau 0.005 \
    --gamma 0.995 \
    --train_freq 1 \
    --gradient_steps 4 \
    --target_update_interval 2 \
    --use_sde \
    --frame_skip 3 \
    --checkpoint_freq 20000

# Optimizations:
# - Large replay buffer for diverse experiences
# - High batch size for stable Q-learning
# - More gradient steps for sample efficiency
# - Higher gamma for long-term planning
```

---

## 📊 Monitoring Training Progress

### Start TensorBoard (in separate terminal)
```bash
# Activate environment first
conda activate peak-rl

# Launch TensorBoard
tensorboard --logdir tensorboard/ --port 6006

# Open browser to: http://localhost:6006
```

### What to Look For:
- **ep_reward_mean**: Should gradually increase (good: >50 after 500k steps)
- **ep_length_mean**: Episodes getting longer means surviving longer
- **fps**: Frames per second (higher is better, aim for >1000)
- **loss**: Should stabilize, not explode
- **learning_rate**: Decreases if using schedule

---

## 🔄 Resume Training from Checkpoint

### Find Your Best Checkpoint
```bash
# List checkpoints
ls -la checkpoints/ppo/
# Look for files like: ppo_peak_500000_steps.zip
```

### Resume PPO Training
```bash
python ppo_training.py \
    --resume checkpoints/ppo/ppo_peak_500000_steps.zip \
    --total_timesteps 3000000 \
    --n_envs 8 \
    --use_sde

# This continues from step 500k to 3M total
```

### Resume SAC with Buffer
```bash
python sac_training.py \
    --resume checkpoints/sac/sac_peak_300000_steps.zip \
    --resume_buffer checkpoints/sac/sac_peak_replay_buffer_300000.pkl \
    --total_timesteps 2000000
```

---

## 🎬 Watch Your Trained Agent

### Test the Best Model
```bash
# Watch PPO agent play
python model_watching.py \
    --algo ppo \
    --model_path best_model/ppo/best_model.zip \
    --num_episodes 5 \
    --video_folder videos/

# Watch SAC agent play
python model_watching.py \
    --algo sac \
    --model_path sac_peak_final.zip \
    --num_episodes 3
```

---

## ⚙️ Hyperparameter Tuning Guide

### PPO Key Parameters

| Parameter | Conservative | Balanced | Aggressive | Effect |
|-----------|-------------|----------|------------|--------|
| learning_rate | 0.0001 | 0.0003 | 0.0007 | How fast agent learns |
| n_envs | 2 | 4-6 | 8-16 | Parallel games (more = faster) |
| batch_size | 32 | 64-128 | 256-512 | Stability vs speed |
| n_epochs | 5 | 10 | 20 | Training intensity |
| ent_coef | 0.001 | 0.01 | 0.05 | Exploration amount |
| clip_range | 0.1 | 0.2 | 0.3 | Update size limit |
| use_sde | False | False | True | Advanced exploration |

### SAC Key Parameters

| Parameter | Conservative | Balanced | Aggressive | Effect |
|-----------|-------------|----------|------------|--------|
| buffer_size | 100k | 500k-1M | 2M+ | Experience memory |
| learning_starts | 1000 | 5000 | 10000 | Exploration phase |
| batch_size | 64 | 256 | 512 | Learning stability |
| tau | 0.001 | 0.005 | 0.01 | Target update speed |
| train_freq | 1 | 2-4 | 8 | Update frequency |
| gradient_steps | 1 | 1-2 | 4 | Updates per step |

---

## 🎯 Training Strategies by Goal

### 🏃 "I Want Results FAST" (2-3 hours)
```bash
python ppo_training.py \
    --n_envs 8 \
    --total_timesteps 500000 \
    --frame_skip 4 \
    --learning_rate 0.0005 \
    --checkpoint_freq 50000
```

### 🏆 "I Want the BEST Model" (8-12 hours)
```bash
# Start with PPO
python ppo_training.py \
    --n_envs 8 \
    --total_timesteps 3000000 \
    --use_sde \
    --use_linear_schedule \
    --checkpoint_freq 25000

# Then try SAC
python sac_training.py \
    --n_envs 6 \
    --total_timesteps 2000000 \
    --buffer_size 1000000 \
    --checkpoint_freq 25000

# Compare both in TensorBoard
```

### 🧪 "I Want to Experiment" (Iterative)
```bash
# Day 1: Baseline
python ppo_training.py --total_timesteps 500000 --save_path models/baseline

# Day 2: Test exploration
python ppo_training.py --total_timesteps 500000 --use_sde --ent_coef 0.05 --save_path models/explore

# Day 3: Test frame skip
python ppo_training.py --total_timesteps 500000 --frame_skip 4 --save_path models/fast

# Compare all models
python model_watching.py --algo ppo --model_path models/baseline.zip
python model_watching.py --algo ppo --model_path models/explore.zip
python model_watching.py --algo ppo --model_path models/fast.zip
```

---

## 🐛 Troubleshooting Common Issues

### "Training is too slow"
```bash
# Solution: Increase parallelization and frame skip
--n_envs 8 --frame_skip 4 --batch_size 256
```

### "Agent gets stuck / doesn't explore"
```bash
# Solution: Increase exploration
--use_sde --ent_coef 0.05 --learning_rate 0.0005
```

### "Training is unstable / reward drops"
```bash
# Solution: More conservative parameters
--learning_rate 0.0001 --clip_range 0.1 --batch_size 64
```

### "Out of GPU memory"
```bash
# Solution: Reduce batch size or environments
--batch_size 32 --n_envs 2
# Or force CPU:
--cpu
```

### "Want to train overnight"
```bash
# Long stable training
python ppo_training.py \
    --total_timesteps 10000000 \
    --checkpoint_freq 100000 \
    --n_envs 4 \
    --frame_skip 2 \
    --learning_rate 0.0001 \
    --use_linear_schedule
```

---

## 📈 Expected Training Timeline

| Steps | Time (GTX 1070) | Expected Performance |
|-------|-----------------|---------------------|
| 100k | 10-15 min | Random actions, learning basics |
| 500k | 1-2 hours | Can climb a bit, falls often |
| 1M | 2-4 hours | Reaches first summit sometimes |
| 2M | 4-8 hours | Consistent first summit, learning level 2 |
| 5M | 10-20 hours | Multiple summits, good performance |

---

## 💾 File Structure After Training

```
Peak-Reninforce-Learning/
├── checkpoints/
│   ├── ppo/
│   │   ├── ppo_peak_50000_steps.zip
│   │   ├── ppo_peak_100000_steps.zip
│   │   └── ...
│   └── sac/
│       └── sac_peak_*.zip
├── best_model/
│   ├── ppo/
│   │   └── best_model.zip
│   └── sac/
│       └── best_model.zip
├── tensorboard/
│   ├── ppo_run_1/
│   └── sac_run_1/
├── logs/
│   ├── ppo/
│   │   └── monitor_*.csv
│   └── sac/
├── videos/
│   └── *.mp4
├── player_template.png
├── summit_template.png
├── ppo_peak_final.zip
└── ppo_peak_final_stats.json
```

---

## 🎓 Learning Path Recommendation

1. **Week 1**: Start with basic PPO, watch it learn
2. **Week 2**: Try intermediate settings, compare results
3. **Week 3**: Experiment with SAC, tune hyperparameters
4. **Week 4**: Combine best settings, train final model

---

## 📝 Quick Reference Card

```bash
# Activate environment (always first!)
conda activate peak-rl

# Quick train
python ppo_training.py --n_envs 4

# Watch progress
tensorboard --logdir tensorboard/

# Test model
python model_watching.py --algo ppo --model_path ppo_peak_final.zip

# Resume training
python ppo_training.py --resume checkpoints/ppo/[latest].zip
```

---

## 🚀 Final Tips

1. **Start simple** - Default parameters first, then optimize
2. **Monitor constantly** - Keep TensorBoard open while training
3. **Save everything** - Checkpoints are your friend
4. **Be patient** - First 100k steps often look random
5. **Experiment** - Try different parameters, compare results
6. **GPU helps** - 5-10x faster than CPU
7. **Frame skip** - Trading accuracy for speed often worth it
8. **Multiple runs** - RL can be random, try multiple seeds

Good luck with your Peak AI training! 🎮🤖