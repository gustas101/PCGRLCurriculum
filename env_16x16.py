import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils import bfs_distance_map, count_connected_components

class DungeonEnv16x16(gym.Env):
    # Tiles: 0: Empty, 1: Wall, 2: Enemy, 3: Key, 4: Door, 5: Player
    def __init__(self, h=16, w=16, obs_size=16, initial_wall_density=0.3):
        super().__init__()
        self.h = h
        self.w = w
        self.active_h = h
        self.active_w = w
        self.obs_size = obs_size
        self._wall_density = initial_wall_density

        # Turtle action space: 0-3 for Move, 4-9 for Tiles
        self.action_space = spaces.Discrete(10)
        self.max_steps = h * w * 2

        # Observation space is ALWAYS fixed to obs_size
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7, self.obs_size, self.obs_size), dtype=np.float32
        )

    @property
    def initial_wall_density(self):
        return self._wall_density

    @initial_wall_density.setter
    def initial_wall_density(self, value):
        self._wall_density = value

    def set_wall_density(self, density):
        """Method callable via env.env_method() to adjust density on the fly."""
        self.initial_wall_density = density

    def set_active_size(self, h, w):
        self.active_h = min(h, self.h)
        self.active_w = min(w, self.w)
        self.max_steps = self.h * self.w * 2

    def get_action_mask(self):
        mask = np.ones(10, dtype=np.int8)
        y, x = self.pos
        if y == 0: mask[0] = 0
        if y >= self.active_h - 1: mask[1] = 0
        if x == 0: mask[2] = 0
        if x >= self.active_w - 1: mask[3] = 0
        if y >= self.active_h or x >= self.active_w or self.pinpoints_mask[y, x] == 1:
            mask[4:] = 0
        return mask

    def _compute_metrics(self, map_):
        n_player = np.sum(map_ == 5)
        n_key = np.sum(map_ == 3)
        n_door = np.sum(map_ == 4)
        n_enemy = np.sum(map_ == 2)
        
        def get_pos(tile_id):
            result = np.where(map_ == tile_id)
            if len(result[0]) > 0:
                return result[0][0], result[1][0]
            return None

        player_pos = get_pos(5)
        key_pos = get_pos(3)
        door_pos = get_pos(4)
        passable = (map_ != 1)
        
        dist_p_k = -1
        if player_pos and key_pos:
            dists = bfs_distance_map(passable, player_pos)
            dist_p_k = dists[key_pos]
            
        dist_k_d = -1
        if key_pos and door_pos:
            dists = bfs_distance_map(passable, key_pos)
            dist_k_d = dists[door_pos]
            
        min_enemy_dist = -1
        if player_pos and n_enemy > 0:
            dists = bfs_distance_map(passable, player_pos)
            enemy_ys, enemy_xs = np.where(map_ == 2)
            valid_dists = [dists[ey, ex] for ey, ex in zip(enemy_ys, enemy_xs) if dists[ey, ex] != -1]
            if valid_dists:
                min_enemy_dist = np.min(valid_dists)
                
        regions = count_connected_components(passable)
        
        return {
            "n_player": n_player, "n_key": n_key, "n_door": n_door, "n_enemy": n_enemy,
            "dist_p_k": dist_p_k, "dist_k_d": dist_k_d, "min_enemy_dist": min_enemy_dist,
            "regions": regions,
            "player_pos": player_pos, "key_pos": key_pos, "door_pos": door_pos
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        h_curr, w_curr = self.active_h, self.active_w
        y_grid, x_grid = np.indices((self.h, self.w))
        map_mask = (y_grid < h_curr) & (x_grid < w_curr)
        
        if self._wall_density >= 1.0:
            random_map = np.ones((self.h, self.w), dtype=np.int32)
        else:
            p_wall = self._wall_density
            p_empty = 1.0 - p_wall
            random_map = self.np_random.choice([0, 1], size=(self.h, self.w), p=[p_empty, p_wall]).astype(np.int32)
        self.map = np.where(map_mask, random_map, 1)

        flat_mask = map_mask.ravel()
        valid_indices = np.where(flat_mask)[0]
        if len(valid_indices) >= 6:
            selected_indices = self.np_random.choice(valid_indices, 6, replace=False)
            flat_map = self.map.ravel()
            flat_map[selected_indices[0]] = 5 # Player
            flat_map[selected_indices[1]] = 3 # Key
            flat_map[selected_indices[2]] = 4 # Door
            flat_map[selected_indices[3]] = 2 # Enemy
            flat_map[selected_indices[4]] = 2 # Enemy
            flat_map[selected_indices[5]] = 2 # Enemy
            self.map = flat_map.reshape((self.h, self.w))
        
        random_pins = self.np_random.choice([0, 1], size=(self.h, self.w), p=[0.9, 0.1]).astype(np.int32)
        self.pinpoints_mask = np.where(map_mask, random_pins, 1)

        for tile_id in [3, 4, 5]:
            ys, xs = np.where(self.map == tile_id)
            for y, x in zip(ys, xs):
                self.pinpoints_mask[y, x] = 1
        
        self.pos = np.array([0, 0], dtype=np.int32)
            
        self.step_count = 0
        self.metrics = self._compute_metrics(self.map)
        return self._get_obs(), {}

    def _get_obs(self):
        pad = self.obs_size // 2
        padded_map_int = np.pad(self.map, pad, constant_values=1)
        oh_padded = np.eye(6)[padded_map_int].transpose(2, 0, 1)
        padded_pins = np.pad(self.pinpoints_mask, pad, constant_values=0)
        y, x = self.pos
        obs_map = oh_padded[:, y : y + self.obs_size, x : x + self.obs_size]
        obs_pins = padded_pins[y : y + self.obs_size, x : x + self.obs_size][np.newaxis, ...]
        return np.concatenate([obs_map, obs_pins], axis=0).astype(np.float32)

    def step(self, action):
        y, x = self.pos
        
        if action < 4:
            if action == 0: y = (y - 1) % self.h
            elif action == 1: y = (y + 1) % self.h
            elif action == 2: x = (x - 1) % self.w
            elif action == 3: x = (x + 1) % self.w
            self.pos = np.array([y, x], dtype=np.int32)
        else:
            tile = action - 4
            if self.pinpoints_mask[y, x] == 0:
                self.map[y, x] = tile

        new_metrics = self._compute_metrics(self.map)
        
        def total_score(m):
            s = 0.0
            s -= 100.0 * (abs(m["n_player"] - 1) + abs(m["n_key"] - 1) + abs(m["n_door"] - 1))
            dist_from_range = max(0, 2 - m["n_enemy"]) + max(0, m["n_enemy"] - 5)
            s -= 20.0 * dist_from_range
            
            if m["dist_p_k"] >= 0: s += 2.0 * m["dist_p_k"]
            else:
                s -= 50.0
                if m["player_pos"] and m["key_pos"]:
                    s -= (abs(m["player_pos"][0]-m["key_pos"][0]) + abs(m["player_pos"][1]-m["key_pos"][1]))
            if m["dist_k_d"] >= 0: s += 2.0 * m["dist_k_d"]
            else:
                s -= 50.0
                if m["key_pos"] and m["door_pos"]:
                    s -= (abs(m["key_pos"][0]-m["door_pos"][0]) + abs(m["key_pos"][1]-m["door_pos"][1]))
            
            if m["min_enemy_dist"] != -1:
                if m["min_enemy_dist"] < 3: s -= 20.0
                elif m["min_enemy_dist"] > 5: s += 5.0
            s -= 10.0 * (m["regions"] - 1)
            return s
            
        reward = total_score(new_metrics) - total_score(self.metrics)
        self.metrics = new_metrics
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        is_solvable = (new_metrics["dist_p_k"] >= 0) and (new_metrics["dist_k_d"] >= 0)
        return self._get_obs(), float(reward), done, False, {"is_success": is_solvable}
