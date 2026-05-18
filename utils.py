import numpy as np
from collections import deque

def bfs_distance_map(passable_grid, start_pos):
    """
    Computes a 2D distance map from start_pos using BFS on passable_grid.
    passable_grid: 2D boolean numpy array (True means passable).
    start_pos: (y, x) tuple.
    Returns: 2D integer numpy array of distances (-1 for unreachable).
    """
    h, w = passable_grid.shape
    dist_map = np.full((h, w), -1, dtype=np.int32)
    
    if not passable_grid[start_pos]:
        return dist_map
        
    dist_map[start_pos] = 0
    queue = deque([start_pos])
    
    while queue:
        cy, cx = queue.popleft()
        current_dist = dist_map[cy, cx]
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and passable_grid[ny, nx]:
                if dist_map[ny, nx] == -1:
                    dist_map[ny, nx] = current_dist + 1
                    queue.append((ny, nx))
                    
    return dist_map

def count_connected_components(passable_grid):
    """
    Calculates the number of isolated passable regions in the grid.
    passable_grid: 2D boolean numpy array.
    Returns: integer.
    """
    h, w = passable_grid.shape
    visited = np.zeros((h, w), dtype=bool)
    count = 0
    
    for y in range(h):
        for x in range(w):
            if passable_grid[y, x] and not visited[y, x]:
                count += 1
                # Start a flood fill (BFS)
                queue = deque([(y, x)])
                visited[y, x] = True
                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and passable_grid[ny, nx]:
                            if not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
    return count
