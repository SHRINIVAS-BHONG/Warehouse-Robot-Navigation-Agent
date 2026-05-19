import numpy as np
from collections import deque

def get_shortest_path(grid_size, start, goal, obstacles):
    """BFS shortest path finder that avoids obstacles."""
    queue = deque([[start]])
    visited = {tuple(start)}
    
    while queue:
        path = queue.popleft()
        curr = path[-1]
        
        if curr == goal:
            return path
            
        x, y = curr
        # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
        neighbors = []
        if y > 0: neighbors.append([x, y-1])
        if y < grid_size - 1: neighbors.append([x, y+1])
        if x > 0: neighbors.append([x-1, y])
        if x < grid_size - 1: neighbors.append([x+1, y])
        
        for n in neighbors:
            if n not in obstacles and tuple(n) not in visited:
                visited.add(tuple(n))
                queue.append(path + [n])
    return None

def get_action_from_states(curr, nxt):
    """Convert a movement from curr to nxt state into the corresponding action index."""
    cx, cy = curr
    nx, ny = nxt
    if ny < cy: return 0  # UP
    if ny > cy: return 1  # DOWN
    if nx < cx: return 2  # LEFT
    if nx > cx: return 3  # RIGHT
    return 0
