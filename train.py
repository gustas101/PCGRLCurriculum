import os
import time
import json
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env import DungeonEnv

# Set default device to CUDA (ROCm) to avoid CPU LAPACK errors during initialization
if torch.cuda.is_available():
    torch.set_default_device("cuda")

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for 8x8 grid that preserves spatial dimensions.
    """
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_obs = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_obs).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

def train_representation(rep_type, total_timesteps=1000000, p_wall=0.3, model_suffix=""):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting {rep_type} training (p_wall={p_wall}) on device: {device}")
    
    # Setup directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    env_kwargs = {
        "h": 8, 
        "w": 8, 
        "obs_size": 8, 
        "representation": rep_type,
        "p_wall": p_wall
    }
    
    # make_vec_env with Monitor wrapper for each sub-process
    env = make_vec_env(
        DungeonEnv, 
        n_envs=16, 
        env_kwargs=env_kwargs, 
        vec_env_cls=SubprocVecEnv,
        monitor_dir="logs/" 
    )
    
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False
    }
    
    model_name = f"{rep_type}{model_suffix}"
    checkpoint_callback = CheckpointCallback(
        save_freq=200000 // 16, 
        save_path='./checkpoints/',
        name_prefix=f'{model_name}_model'
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log="./ppo_dungeon_tensorboard/"
    )
    
    start_wall_time = time.time()
    model.learn(
        total_timesteps=total_timesteps, 
        tb_log_name=f"{model_name}_run",
        callback=checkpoint_callback
    )
    end_wall_time = time.time()
    
    training_duration = end_wall_time - start_wall_time
    
    # Save training time
    times_file = "training_times.json"
    if os.path.exists(times_file):
        with open(times_file, "r") as f:
            times_data = json.load(f)
    else:
        times_data = {}
    
    times_data[model_name] = training_duration
    with open(times_file, "w") as f:
        json.dump(times_data, f, indent=4)
    
    model.save(f"models/{model_name}_ppo_dungeon")
    print(f"Training for {model_name} complete in {training_duration:.2f}s! Model saved.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        rep = sys.argv[1]
        p_wall = 0.3
        suffix = ""
        if len(sys.argv) > 2:
            p_wall = float(sys.argv[2])
        if len(sys.argv) > 3:
            suffix = sys.argv[3]
        train_representation(rep, p_wall=p_wall, model_suffix=suffix)
    else:
        print("Usage: python train.py <rep> [p_wall] [suffix]")

