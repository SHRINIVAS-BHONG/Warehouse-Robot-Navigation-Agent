import random
import numpy as np

class MultiAgentWarehouseEnv:
    """Phase 8 (Step 31): Multi-Agent Navigation.
    Supports 2 robots operating simultaneously on the same grid.
    They must avoid each other while navigating to their individual goals.
    """
    def __init__(self, grid_size=10, num_obstacles=8):
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.action_space = list(range(4)) # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        
        # State: r1_x, r1_y, r2_x, r2_y, g1_x, g1_y, g2_x, g2_y
        self.state_dim = 8

    def reset(self, seed=None, options=None):
        # Spawn Robot 1 and Goal 1
        self.r1_pos = [0, 0]
        self.g1_pos = [self.grid_size - 1, self.grid_size - 1]
        
        # Spawn Robot 2 and Goal 2
        self.r2_pos = [self.grid_size - 1, 0]
        self.g2_pos = [0, self.grid_size - 1]
        
        # Spawn Obstacles
        self.obstacles = []
        while len(self.obstacles) < self.num_obstacles:
            o = [random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)]
            if o not in [self.r1_pos, self.r2_pos, self.g1_pos, self.g2_pos] and o not in self.obstacles:
                self.obstacles.append(o)
                
        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.r1_pos[0], self.r1_pos[1],
            self.r2_pos[0], self.r2_pos[1],
            self.g1_pos[0], self.g1_pos[1],
            self.g2_pos[0], self.g2_pos[1]
        ], dtype=np.float32)

    def step(self, actions):
        """actions is a tuple/list of two actions: [action_robot1, action_robot2]"""
        a1, a2 = actions
        
        # Calculate intended new positions
        new_r1 = self._get_new_pos(self.r1_pos, a1)
        new_r2 = self._get_new_pos(self.r2_pos, a2)
        
        # Collision resolution
        r1_collides = new_r1 in self.obstacles
        r2_collides = new_r2 in self.obstacles
        
        # Check if robots hit each other (same cell, or swapped cells)
        robots_collide = new_r1 == new_r2 or (new_r1 == self.r2_pos and new_r2 == self.r1_pos)
        
        r1_reward, r2_reward = -0.1, -0.1
        
        if r1_collides or robots_collide:
            r1_reward = -50
        else:
            self.r1_pos = new_r1
            
        if r2_collides or robots_collide:
            r2_reward = -50
        else:
            self.r2_pos = new_r2
            
        # Goal checks
        d1, d2 = False, False
        if self.r1_pos == self.g1_pos:
            r1_reward = 100
            d1 = True
            
        if self.r2_pos == self.g2_pos:
            r2_reward = 100
            d2 = True
            
        done = d1 and d2
        
        return self._get_state(), (r1_reward, r2_reward), done, False, {}

    def _get_new_pos(self, pos, action):
        x, y = pos
        if action == 0: y -= 1
        elif action == 1: y += 1
        elif action == 2: x -= 1
        elif action == 3: x += 1
        x = int(np.clip(x, 0, self.grid_size - 1))
        y = int(np.clip(y, 0, self.grid_size - 1))
        return [x, y]

    def render(self):
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        for ox, oy in self.obstacles:
            grid[oy][ox] = "X"
            
        grid[self.g1_pos[1]][self.g1_pos[0]] = "G1"
        grid[self.g2_pos[1]][self.g2_pos[0]] = "G2"
        grid[self.r1_pos[1]][self.r1_pos[0]] = "R1"
        grid[self.r2_pos[1]][self.r2_pos[0]] = "R2"
        
        print("Multi-Agent Warehouse View:")
        for row in grid:
            # Format nicely for 2 chars
            print(" ".join([cell.ljust(2) for cell in row]))
        print()
