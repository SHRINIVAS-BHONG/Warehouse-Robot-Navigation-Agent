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
        while len(self.obstacles) < num_obstacles:
            obstacle = [rng.randint(0, self.grid_size - 1), rng.randint(0, self.grid_size - 1)]
            if (
                obstacle != [0, 0]  # Start pos
                and obstacle != self.goal_pos
                and obstacle not in self.obstacles
            ):
                self.obstacles.append(obstacle)
                
        print("Warehouse Environment Initialized")
        print("Action Space:", self.action_space)
        print("Observation Space:", self.observation_space)

    def reset(self, seed=None, options=None):
        # Start robot at top‑left corner
        self.robot_pos = [0, 0]
        return np.array(self.robot_pos), {}

    def step(self, action):
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
        reward = -0.1
        done = False
        if new_position in self.obstacles:
            reward = -100
        else:
            self.robot_pos = new_position
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
