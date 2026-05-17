import numpy as np
from env.advanced_warehouse_env import AdvancedWarehouseEnv

class PartialObsWarehouseEnv(AdvancedWarehouseEnv):
    """Phase 6: Partial Observability.
    Robot only sees a 5x5 local sensor window around itself.
    Requires inferring environment from limited information.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_range = 2 # 2 cells in each direction = 5x5 window
        # 5x5 grid + dx + dy + battery = 25 + 1 + 1 + 1 = 28 dim state
        self.state_dim = (2 * self.sensor_range + 1)**2 + 3

    def _get_state(self):
        rx, ry = self.robot_pos
        
        # Construct Local Window 5x5
        window_size = 2 * self.sensor_range + 1
        local_grid = np.zeros((window_size, window_size), dtype=np.float32)
        
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                vx, vy = rx + dx, ry + dy
                grid_y = dy + self.sensor_range
                grid_x = dx + self.sensor_range
                
                if vx < 0 or vx >= self.grid_size or vy < 0 or vy >= self.grid_size:
                    local_grid[grid_y][grid_x] = -1 # Wall / Out of bounds
                elif [vx, vy] in self.obstacles:
                    local_grid[grid_y][grid_x] = 1  # Obstacle
                elif self.current_goal_idx < len(self.goals) and [vx, vy] == self.goals[self.current_goal_idx]:
                    local_grid[grid_y][grid_x] = 2  # Goal
                else:
                    local_grid[grid_y][grid_x] = 0  # Empty space
                    
        local_flat = local_grid.flatten()
        
        # Calculate goal relative displacement (compass)
        if self.current_goal_idx < len(self.goals):
            gx, gy = self.goals[self.current_goal_idx]
            dir_x = gx - rx
            dir_y = gy - ry
        else:
            dir_x, dir_y = 0, 0
            
        return np.concatenate([local_flat, [dir_x, dir_y, self.battery]]).astype(np.float32)

    def render(self):
        """Render both global ground-truth and local sensor view."""
        print("=== GLOBAL GROUND TRUTH ===")
        super().render()
        
        print("=== LOCAL 5x5 SENSOR VIEW ===")
        state = self._get_state()
        local_flat = state[:25]
        dir_x, dir_y, bat = state[25:]
        
        idx = 0
        for _ in range(5):
            row = []
            for _ in range(5):
                val = local_flat[idx]
                if val == -1: row.append("#")   # Wall
                elif val == 1: row.append("X")  # Obstacle
                elif val == 2: row.append("G")  # Goal
                elif idx == 12: row.append("R") # Center is always Robot
                else: row.append(".")
                idx += 1
            print(" ".join(row))
            
        print(f"Goal Compass (dx, dy): ({dir_x}, {dir_y}) | Battery: {bat}\n")
