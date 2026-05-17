import random
import numpy as np

class AdvancedWarehouseEnv:
    """Advanced Warehouse Environment (Phase 3)
    Features: 
    - Dynamic obstacles: Move randomly each step.
    - Multiple goals: Robot must navigate to sequential pickup stations.
    - Battery constraints: Finite energy budget for navigation.
    - Randomized layouts: Different layouts per episode.
    """
    def __init__(self, grid_size=10, num_obstacles=10, num_goals=3, max_battery=150):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals
        self.max_battery = max_battery
        
        self.action_space = list(range(4))  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        
        # State representation (1D Vector):
        # [robot_x, robot_y, current_goal_x, current_goal_y, battery_level, obs1_x, obs1_y, ...]
        self.state_dim = 5 + 2 * self.num_obstacles

    def reset(self, seed=None, options=None):
        self.battery = self.max_battery
        self.current_goal_idx = 0
        
        # Randomize multiple goals (Pickup stations)
        self.goals = []
        while len(self.goals) < self.num_goals:
            g = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if g not in self.goals:
                self.goals.append(g)
                
        # Randomize robot spawn location
        while True:
            r = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if r not in self.goals:
                self.robot_pos = r
                break
                
        # Randomize obstacle spawn locations
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            o = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if o != self.robot_pos and o not in self.goals and o not in self.obstacles:
                self.obstacles.append(o)
                
        return self._get_state(), {}

    def _get_state(self):
        # If all goals are reached, point to a dummy goal (-1, -1)
        if self.current_goal_idx < len(self.goals):
            gx, gy = self.goals[self.current_goal_idx]
        else:
            gx, gy = -1, -1
            
        state = [self.robot_pos[0], self.robot_pos[1], gx, gy, self.battery]
        for obs in self.obstacles:
            state.extend(obs)
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # Movement penalty & battery drain
        self.battery -= 1
        reward = -0.1
        done = False
        
        x, y = self.robot_pos
        if action == 0:   # UP
            y -= 1
        elif action == 1: # DOWN
            y += 1
        elif action == 2: # LEFT
            x -= 1
        elif action == 3: # RIGHT
            x += 1
        
        # Boundary constraints
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        new_pos = [x, y]
        
        # Obstacle collision check
        if new_pos in self.obstacles:
            reward = -100
        else:
            self.robot_pos = new_pos
            
        # Goal reached check
        if self.current_goal_idx < len(self.goals) and self.robot_pos == self.goals[self.current_goal_idx]:
            reward += 50  # Intermediate goal reward
            self.current_goal_idx += 1
            if self.current_goal_idx >= len(self.goals):
                reward += 100  # Final goal bonus
                done = True
                
        # Battery constraint check
        if self.battery <= 0 and not done:
            reward = -50  # Dead battery penalty
            done = True
            
        # Dynamic obstacles move automatically
        self._move_obstacles()
        
        info = {
            "battery": self.battery,
            "goals_reached": self.current_goal_idx
        }
        
        return self._get_state(), reward, done, False, info

    def _move_obstacles(self):
        """Randomly move obstacles around the grid."""
        for i in range(len(self.obstacles)):
            ox, oy = self.obstacles[i]
            # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:STAY
            move = random.randint(0, 4)
            nx, ny = ox, oy
            if move == 0: ny -= 1
            elif move == 1: ny += 1
            elif move == 2: nx -= 1
            elif move == 3: nx += 1
            
            # Ensure within grid
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                new_obs = [nx, ny]
                # Avoid moving into goals, the robot, or other obstacles
                if new_obs != self.robot_pos and new_obs not in self.goals and new_obs not in self.obstacles:
                    self.obstacles[i] = new_obs

    def render(self):
        """Render the grid as ASCII art."""
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # Display pending goals
        for g_idx, g in enumerate(self.goals):
            if g_idx >= self.current_goal_idx:
                grid[g[1]][g[0]] = "G"
                
        # Display dynamic obstacles
        for ox, oy in self.obstacles:
            grid[oy][ox] = "X"
            
        # Display robot
        rx, ry = self.robot_pos
        grid[ry][rx] = "R"
        
        print(f"Battery: {self.battery}/{self.max_battery} | Packages Collected: {self.current_goal_idx}/{self.num_goals}")
        for row in grid:
            print(" ".join(row))
        print()
