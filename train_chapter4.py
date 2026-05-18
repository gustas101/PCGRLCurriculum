import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from env_16x16 import DungeonEnv16x16
from curriculum_callback import CurriculumManager

# Set default device to CUDA (ROCm) to avoid CPU LAPACK errors during initialization
if torch.cuda.is_available():
    torch.set_default_device("cuda")

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN for 16x16 grid that preserves spatial dimensions initially, 
    matching the architecture from Chapter 3 but dynamically scaling the flattened output.
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting Chapter 4 Curriculum Training (16x16) on device: {device}")
    
    # Setup directories
    os.makedirs("logs_chap4", exist_ok=True)
    os.makedirs("checkpoints_chap4", exist_ok=True)
    os.makedirs("models_chap4", exist_ok=True)
    
    # 1. Initialize Vectorized Environment
    env_kwargs = {
        "h": 16, 
        "w": 16, 
        "obs_size": 16,  # 16x16 full observation
        "initial_wall_density": 0.30
    }
    
    num_envs = 16
    env = make_vec_env(
        DungeonEnv16x16, 
        n_envs=num_envs, 
        env_kwargs=env_kwargs, 
        vec_env_cls=SubprocVecEnv,
        monitor_dir="logs_chap4/",
        monitor_kwargs={"info_keywords": ("is_success",)}
    )
    
    # 2. Setup Architecture & Agent
    policy_kwargs = {
        "features_extractor_class": CustomCNN,
        "features_extractor_kwargs": {"features_dim": 512},
        "normalize_images": False
    }
    
    # 3. Setup Callbacks (Curriculum & Checkpoints)
    # Track the percentage of maps that are solvable.
    # When 85% of maps are solvable, increase the density.
    curriculum_callback = CurriculumManager(
        check_freq=10000, 
        success_threshold=0.85, 
        density_step=0.05, 
        max_density=1.0, 
        save_dir="./checkpoints_chap4/"
    )
    
    # Save a general checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=500000 // num_envs, 
        save_path='./checkpoints_chap4/',
        name_prefix='turtle_16x16_model'
    )
    
    callbacks = CallbackList([curriculum_callback, checkpoint_callback])
    
    print("Initializing PPO...")
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        policy_kwargs=policy_kwargs,
        device=device,
        tensorboard_log="./tensorboard_chap4/"
    )
    
    # 4. Train the Model for 10M Timesteps
    print("Starting training for 10,000,000 timesteps...")
    model.learn(
        total_timesteps=10_000_000, 
        tb_log_name="turtle_curriculum_run",
        callback=callbacks
    )
    
    # Save final model
    model.save("models_chap4/turtle_16x16_curriculum_final")
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    main()
