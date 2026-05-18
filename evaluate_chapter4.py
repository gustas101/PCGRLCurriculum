import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env_16x16 import DungeonEnv16x16

# Ensure directories exist
os.makedirs("results_chap4", exist_ok=True)

# Checkpoint definitions
CHECKPOINTS = [
    {
        "name": "Density 35%",
        "density": 0.35,
        "path": "checkpoints_chap4/ppo_curriculum_density_035_steps_640000.zip"
    },
    {
        "name": "Density 50%",
        "density": 0.49,
        "path": "checkpoints_chap4/ppo_curriculum_density_049_steps_1120000.zip"
    },
    {
        "name": "Density 65%",
        "density": 0.65,
        "path": "checkpoints_chap4/turtle_16x16_model_2500000_steps.zip"
    }
]

# Standard color mapping for tiles
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
custom_colors = ['#F0F0F0', '#2F4F4F', '#DC143C', '#FFD700', '#8B4513', '#1E90FF']
cmap = ListedColormap(custom_colors)

tile_labels = {
    'Empty': '#F0F0F0', 'Wall': '#2F4F4F', 'Enemy': '#DC143C',
    'Key': '#FFD700', 'Door': '#8B4513', 'Player': '#1E90FF'
}

def evaluate_models(num_episodes=100):
    results = {}
    sample_grids = {cp["name"]: [] for cp in CHECKPOINTS}
    
    # Initialize environment
    env = DungeonEnv16x16(h=16, w=16, obs_size=16)
    
    for cp in CHECKPOINTS:
        print(f"Evaluating {cp['name']} from {cp['path']}...")
        if not os.path.exists(cp['path']):
            print(f"Warning: Model path {cp['path']} not found. Skipping.")
            continue
            
        model = PPO.load(cp['path'])
        env.initial_wall_density = cp['density']
        
        metrics_tracker = {
            "solvable_count": 0,
            "path_lengths": [],
            "change_pcts": [],
            "regions": [],
            "enemies": [],
            "unique_maps": set()
        }
        
        for ep in range(num_episodes):
            obs, _ = env.reset()
            initial_map = env.map.copy()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, done, truncated, info = env.step(action)
            
            # Calculate metrics
            final_map = env.map.copy()
            map_tuple = tuple(final_map.flatten())
            metrics_tracker["unique_maps"].add(map_tuple)
            
            # Change percentage
            changed_tiles = np.sum(initial_map != final_map)
            change_pct = (changed_tiles / (16 * 16)) * 100
            metrics_tracker["change_pcts"].append(change_pct)
            
            # Use environment's computed metrics
            metrics = env.metrics
            metrics_tracker["regions"].append(metrics["regions"])
            metrics_tracker["enemies"].append(metrics["n_enemy"])
            
            # Solvability and Path Length
            is_solvable = info.get("is_success", False)
            if is_solvable:
                metrics_tracker["solvable_count"] += 1
                total_path = metrics["dist_p_k"] + metrics["dist_k_d"]
                metrics_tracker["path_lengths"].append(total_path)
                
                # Save first 3 solvable grids for visualization
                if len(sample_grids[cp["name"]]) < 3:
                    sample_grids[cp["name"]].append(final_map.copy())

        # Compile final stats for this model
        solvability_rate = (metrics_tracker["solvable_count"] / num_episodes) * 100
        avg_path = np.mean(metrics_tracker["path_lengths"]) if metrics_tracker["path_lengths"] else 0
        avg_change = np.mean(metrics_tracker["change_pcts"])
        avg_regions = np.mean(metrics_tracker["regions"])
        avg_enemies = np.mean(metrics_tracker["enemies"])
        
        results[cp["name"]] = {
            "Density": cp["density"],
            "Solvability Rate (%)": solvability_rate,
            "Avg Path Length (Solvable Only)": avg_path,
            "Avg Change Percentage (%)": avg_change,
            "Avg Regions": avg_regions,
            "Avg Enemies": avg_enemies,
            "Unique Maps Count": len(metrics_tracker["unique_maps"])
        }
        
        print(f"  -> Solvability: {solvability_rate}% | Avg Path: {avg_path:.2f} | Unique Maps: {len(metrics_tracker['unique_maps'])}")

    # Save JSON Report
    json_path = "results_chap4/chapter4_eval.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation metrics saved to {json_path}")
    
    return sample_grids

def generate_visualization(sample_grids):
    print("Generating 3x3 Visualization Matrix...")
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Chapter 4: Curriculum Evolution (16x16)", fontsize=16, y=0.95)
    
    for row_idx, cp in enumerate(CHECKPOINTS):
        name = cp["name"]
        grids = sample_grids.get(name, [])
        
        # Add row title
        axes[row_idx, 0].set_ylabel(name, fontsize=14, labelpad=10)
        
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            if col_idx < len(grids):
                ax.imshow(grids[col_idx], cmap=cmap, vmin=0, vmax=5)
            else:
                # Placeholder if < 3 solvable grids found
                ax.text(0.5, 0.5, 'No Solvable\nMap Generated', 
                        horizontalalignment='center', verticalalignment='center', 
                        transform=ax.transAxes, color='gray')
                ax.set_facecolor('#f0f0f0')
            
            # Remove ticks for clean academic look
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, left=0.1)
    
    # Add shared legend
    legend_elements = [Patch(facecolor=color, edgecolor='gray', label=label) for label, color in tile_labels.items()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=12)
    plt.subplots_adjust(bottom=0.1)
    
    img_path = "results_chap4/figure_19_chapter4_evolution.png"
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {img_path}")
    plt.close()

if __name__ == "__main__":
    grids = evaluate_models(num_episodes=100)
    generate_visualization(grids)
