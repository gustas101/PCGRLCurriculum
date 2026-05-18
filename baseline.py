import numpy as np
import random

def generate_baseline_level(seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    grid = np.ones((8, 8), dtype=int)
    
    # Simple randomized DFS to carve a path
    def get_neighbors(r, c):
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 8 and 0 <= nc < 8:
                neighbors.append((nr, nc))
        random.shuffle(neighbors)
        return neighbors

    start_r, start_c = random.randint(0, 7), random.randint(0, 7)
    path = [(start_r, start_c)]
    grid[start_r, start_c] = 0
    visited = set([(start_r, start_c)])
    
    # Let's carve a path of length at least 15 if possible
    target_length = random.randint(15, 25)
    
    current_r, current_c = start_r, start_c
    while len(path) < target_length:
        moved = False
        for nr, nc in get_neighbors(current_r, current_c):
            if (nr, nc) not in visited:
                # Check if it connects to other visited cells (we want a single path without too many loops)
                visited_neighbors = sum(1 for nnr, nnc in get_neighbors(nr, nc) if (nnr, nnc) in visited)
                if visited_neighbors == 1:
                    visited.add((nr, nc))
                    path.append((nr, nc))
                    grid[nr, nc] = 0
                    current_r, current_c = nr, nc
                    moved = True
                    break
        if not moved:
            # Backtrack or stop
            break
            
    # Need at least 3 tiles for Player, Key, Door
    if len(path) < 3:
        return generate_baseline_level()
        
    # Place Player at start, Door at end
    player_pos = path[0]
    door_pos = path[-1]
    
    # Key exactly halfway
    key_idx = len(path) // 2
    key_pos = path[key_idx]
    
    grid[player_pos[0], player_pos[1]] = 5 # Player
    grid[key_pos[0], key_pos[1]] = 3 # Key
    grid[door_pos[0], door_pos[1]] = 4 # Door
    
    # Place exactly 3 Enemies on non-path tiles
    non_path_tiles = []
    for r in range(8):
        for c in range(8):
            if (r, c) not in path:
                non_path_tiles.append((r, c))
                
    if len(non_path_tiles) >= 3:
        enemy_positions = random.sample(non_path_tiles, 3)
        for r, c in enemy_positions:
            grid[r, c] = 2 # Enemy
            
    return grid

if __name__ == "__main__":
    grid = generate_baseline_level()
    print(grid)