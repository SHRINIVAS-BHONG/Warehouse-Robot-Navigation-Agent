import random
import numpy as np

class WarehouseEnv:
    """A minimal grid world environment.
    Robot starts at (0,0), goal at bottom‑right, with random obstacles.
    Rendering is done with simple ASCII art to avoid external graphics dependencies.
    """
    def __init__(self):
        self.grid_size = 10
        self.action_space = list(range(4))  # 0=UP,1=DOWN,2=LEFT,3=RIGHT
        self.observation_space = (0, self.grid_size - 1)
        
        # Initialize static obstacles for early phases
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.obstacles = []
        num_obstacles = 10
        # Set a fixed seed for reproducible obstacle layout
        rng = random.Random(42)
        
        # Define positions adjacent to start and goal that must be kept clear
        start_pos = [0, 0]
        adjacent_to_start = [[1, 0], [0, 1]]  # Right and down from start
        adjacent_to_goal = [[self.grid_size - 2, self.grid_size - 1], 
                           [self.grid_size - 1, self.grid_size - 2]]  # Left and up from goal
        blocked_positions = [start_pos, self.goal_pos] + adjacent_to_start + adjacent_to_goal
        
        while len(self.obstacles) < num_obstacles:
            obstacle = [rng.randint(0, self.grid_size - 1), rng.randint(0, self.grid_size - 1)]
            if obstacle not in blocked_positions and obstacle not in self.obstacles:
                self.obstacles.append(obstacle)
                
        print("Warehouse Environment Initialized")
        print("Action Space:", self.action_space)
        print("Observation Space:", self.observation_space)

    def reset(self, seed=None, options=None):
        # Start robot at top‑left corner
        self.robot_pos = [0, 0]
        return np.array(self.robot_pos), {}

    def step(self, action):
        old_x, old_y = self.robot_pos
        x, y = self.robot_pos
        if action == 0:   # UP
            y -= 1
        elif action == 1: # DOWN
            y += 1
        elif action == 2: # LEFT
            x -= 1
        elif action == 3: # RIGHT
            x += 1
        # Keep inside grid
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        new_position = [x, y]
        
        # Calculate old and new distances to goal for reward shaping
        old_distance = abs(old_x - self.goal_pos[0]) + abs(old_y - self.goal_pos[1])
        new_distance = abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1])
        
        reward = -0.1
        done = False
        
        # Check if hit obstacle
        if new_position in self.obstacles:
            reward = -10  # Reduced from -100 to allow learning
            # Don't move if obstacle
        else:
            self.robot_pos = new_position
            # Reward shaping: encourage moving closer to goal
            if new_distance < old_distance:
                reward = 1.0  # Positive reward for getting closer
            elif new_distance > old_distance:
                reward = -1.0  # Penalty for moving away
            else:
                reward = -0.1  # Small penalty for not making progress
                
        # Check if reached goal
        if self.robot_pos == self.goal_pos:
            reward = 100
            done = True
            
        return np.array(self.robot_pos), reward, done, False, {}

    def render(self):
        """Render the grid as ASCII art.
        R = robot, G = goal, X = obstacle, . = empty cell.
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for ox, oy in self.obstacles:
            grid[oy][ox] = "X"
        gx, gy = self.goal_pos
        grid[gy][gx] = "G"
        rx, ry = self.robot_pos
        grid[ry][rx] = "R"
        for row in grid:
            print(" ".join(row))
        print()
