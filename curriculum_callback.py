import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumManager(BaseCallback):
    """
    Custom callback for Curriculum Learning.
    Monitors rolling average episode rewards and increases the wall density
    of the environment when the threshold is crossed.
    """
    def __init__(self, check_freq=10000, success_threshold=0.85, density_step=0.05, max_density=1.0, verbose=1, save_dir=None):
        super(CurriculumManager, self).__init__(verbose)
        self.check_freq = check_freq
        self.success_threshold = success_threshold
        self.density_step = density_step
        self.max_density = max_density
        self.save_dir = save_dir
        self.current_density = 0.30
        
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check every N steps
        if self.n_calls % self.check_freq == 0:
            if len(self.model.ep_info_buffer) > 0:
                # Calculate mean success rate from the rolling buffer
                success_rates = [ep_info['is_success'] for ep_info in self.model.ep_info_buffer if 'is_success' in ep_info]
                
                if len(success_rates) > 0:
                    success_rate = np.mean(success_rates)
                    self.logger.record('curriculum/success_rate', success_rate)
                    
                    # Check threshold
                    if success_rate >= self.success_threshold and self.current_density < self.max_density:
                        self.current_density = min(self.max_density, self.current_density + self.density_step)
                        
                        if self.verbose > 0:
                            print(f"\n[CurriculumManager] Step {self.num_timesteps}: Success Rate {success_rate:.2f} >= {self.success_threshold}.")
                            print(f"-> Increasing wall density to {self.current_density:.2f}")
                        
                        # Apply new density to all vectorized environments
                        self.training_env.env_method('set_wall_density', self.current_density)
                        
                        # Save model upon curriculum tier advancement
                        if self.save_dir is not None:
                            tier_density_str = int(self.current_density * 100)
                            save_path = os.path.join(self.save_dir, f"ppo_curriculum_density_{tier_density_str:03d}_steps_{self.num_timesteps}")
                            self.model.save(save_path)
                            if self.verbose > 0:
                                print(f"Saved curriculum milestone model to {save_path}")

        # Always log the density so it can be graphed
        self.logger.record('curriculum/wall_density', self.current_density)
        
        return True
