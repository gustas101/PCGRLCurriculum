import json
import time
import os
import pandas as pd
import numpy as np
import torch
from stable_baselines3 import PPO
from env import DungeonEnv
from baseline import generate_baseline_level

# Set default device to CUDA (ROCm) to avoid CPU LAPACK errors
if torch.cuda.is_available():
    torch.set_default_device("cuda")

def tile_to_char(tile):
    # Tiles: 0: Empty, 1: Wall, 2: Enemy, 3: Key, 4: Door, 5: Player
    chars = {0: '.', 1: '#', 2: 'E', 3: 'K', 4: 'D', 5: 'P'}
    return chars.get(tile, '?')

def grid_to_string(grid):
    return '\n'.join([''.join([tile_to_char(t) for t in row]) for row in grid])

def evaluate_rl_agent(representation, model_path, num_levels=300, p_wall=0.3):
    env_kwargs = {"h": 8, "w": 8, "obs_size": 8, "representation": representation, "p_wall": p_wall}
    env = DungeonEnv(**env_kwargs)
    
    try:
        model = PPO.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Warning: {model_path} not found. Using untrained PPO for {representation}.")
        model = PPO("CnnPolicy", env, verbose=0)
    
    metrics_list = []
    all_grids = []
    
    start_time = time.time()
    for i in range(num_levels):
        obs, _ = env.reset()
        initial_map = env.map.copy()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        final_metrics = env._compute_metrics(env.map)
        final_map = env.map.copy()
        
        # Calculate solvability
        is_solvable = (final_metrics["dist_p_k"] != -1 and final_metrics["dist_k_d"] != -1)
        path_length = 0
        if is_solvable:
            path_length = final_metrics["dist_p_k"] + final_metrics["dist_k_d"]
            
        # Change percentage
        changes = np.sum(initial_map != final_map)
        change_pct = (changes / (env.h * env.w)) * 100
        
        metrics_list.append({
            "solvability": 1.0 if is_solvable else 0.0,
            "path_length": path_length,
            "change_pct": change_pct,
            "enemies": final_metrics["n_enemy"],
            "regions": final_metrics["regions"],
            "grid_str": grid_to_string(final_map)
        })
        all_grids.append(grid_to_string(final_map))
            
    total_time = time.time() - start_time
    avg_time = total_time / num_levels
    
    summary = aggregate_results(metrics_list, avg_time)
    return summary, all_grids

def evaluate_baseline(num_levels=300):
    env = DungeonEnv(h=8, w=8, obs_size=8) # Just for metric calculation
    metrics_list = []
    all_grids = []
    
    start_time = time.time()
    for i in range(num_levels):
        # We assume baseline starts from all walls or empty, but let's just 
        # compare it to an all-wall grid to see "changes" if needed, 
        # or just 0 for baseline.
        initial_map = np.ones((8, 8), dtype=int) 
        
        final_map = generate_baseline_level()
        final_metrics = env._compute_metrics(final_map)
        
        is_solvable = (final_metrics["dist_p_k"] != -1 and final_metrics["dist_k_d"] != -1)
        path_length = 0
        if is_solvable:
            path_length = final_metrics["dist_p_k"] + final_metrics["dist_k_d"]
            
        changes = np.sum(initial_map != final_map)
        change_pct = (changes / 64) * 100
        
        metrics_list.append({
            "solvability": 1.0 if is_solvable else 0.0,
            "path_length": path_length,
            "change_pct": change_pct,
            "enemies": final_metrics["n_enemy"],
            "regions": final_metrics["regions"],
            "grid_str": grid_to_string(final_map)
        })
        all_grids.append(grid_to_string(final_map))
            
    total_time = time.time() - start_time
    avg_time = total_time / num_levels
    
    summary = aggregate_results(metrics_list, avg_time)
    return summary, all_grids

def aggregate_results(metrics_list, avg_time):
    solvability_rates = [m['solvability'] for m in metrics_list]
    solvable_paths = [m['path_length'] for m in metrics_list if m['solvability'] > 0]
    change_pcts = [m['change_pct'] for m in metrics_list]
    grid_strs = [m['grid_str'] for m in metrics_list]
    
    unique_maps = len(set(grid_strs))
    
    return {
        "Solvability Rate (%)": np.mean(solvability_rates) * 100,
        "Avg Path Length (Solvable Only)": np.mean(solvable_paths) if solvable_paths else 0,
        "Avg Change Percentage (%)": np.mean(change_pcts),
        "Unique Maps Count": unique_maps,
        "Avg Enemies": np.mean([m['enemies'] for m in metrics_list]),
        "Avg Regions": np.mean([m['regions'] for m in metrics_list]),
        "Generation Time (s)": avg_time
    }

if __name__ == "__main__":
    results = {}
    all_method_grids = {}
    
    # 1. Baseline
    print("Evaluating Baseline...")
    res, grids = evaluate_baseline()
    results["Baseline"] = res
    all_method_grids["Baseline"] = grids
    
    # 2. RL Agents
    agents = [
        ("narrow", "models/narrow_ppo_dungeon"),
        ("turtle", "models/turtle_ppo_dungeon"),
        ("wide", "models/wide_ppo_dungeon")
    ]
    
    for rep, path in agents:
        print(f"Evaluating {rep}...")
        res, grids = evaluate_rl_agent(rep, path)
        results[rep] = res
        all_method_grids[rep] = grids
        
    # Save Reports
    df = pd.DataFrame(results).T
    df.to_csv("evaluation_report_v2.csv")
    with open("evaluation_report_v2.json", "w") as f:
        json.dump(results, f, indent=4)
        
    # Save ALL 100 Grids for each
    with open("all_generated_grids.txt", "w") as f:
        for method, grids in all_method_grids.items():
            f.write(f"########################################\n")
            f.write(f"### METHOD: {method}\n")
            f.write(f"########################################\n\n")
            for idx, grid in enumerate(grids):
                f.write(f"--- Map {idx+1} ---\n{grid}\n\n")
                
    print("Evaluation complete! Reports and all grids saved.")
