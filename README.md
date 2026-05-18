# PCGRL Level Generation Evaluation Suite

This repository contains the source code for a Procedural Content Generation via Reinforcement Learning (PCGRL) framework, focusing on generating solvable 2D dungeon levels. 

## Project Overview
The suite evaluates different Reinforcement Learning (RL) agent representations (Narrow, Turtle, Wide) on their ability to create 8x8 and 16x16 dungeon levels that satisfy solvability constraints (Player -> Key -> Door path) and secondary objectives (enemy placement, map preservation).

### Key Features
- **Custom Gymnasium Environments**: Supports 8x8 and 16x16 grid scales.
- **Agent Representations**: Implementation of Narrow, Turtle, and Wide editing policies.
- **Curriculum Learning**: Dynamic difficulty adjustment (increasing wall density) for training agents in high-density environments.
- **Stable Baselines 3 Integration**: Uses PPO (Proximal Policy Optimization) for robust training.
- **Academic Metrics**: Automatic calculation of Solvability Rate, Path Length, Change Percentage, and Connected Components.

## File Structure
- `env.py`: The core 8x8 PCGRL environment.
- `env_16x16.py`: Scaled 16x16 environment supporting dynamic wall density.
- `utils.py`: Pathfinding (BFS) and topological analysis utilities.
- `train.py`: Main training script for 8x8 models.
- `train_chapter4.py`: Curriculum training script for 16x16 models.
- `curriculum_callback.py`: Custom SB3 callback for automated curriculum advancement.
- `evaluate.py`: Standard evaluation suite for 8x8 models.
- `evaluate_chapter4.py`: Evaluation suite for curriculum-trained 16x16 models.
- `baseline.py`: Procedural algorithmic baseline (Randomized DFS).
- `requirements.txt`: Python dependencies.

## Installation
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
```

## Usage

### Training an 8x8 Agent
```bash
python train.py turtle 0.3
```

### Training with Curriculum (16x16)
```bash
python train_chapter4.py
```

### Evaluation
To evaluate 8x8 models:
```bash
python evaluate.py
```
To evaluate 16x16 curriculum models:
```bash
python evaluate_chapter4.py
```
