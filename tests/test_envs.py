import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.warehouse_env import WarehouseEnv
from env.advanced_warehouse_env import AdvancedWarehouseEnv
from env.partial_obs_env import PartialObsWarehouseEnv
from env.multi_agent_env import MultiAgentWarehouseEnv

def test_warehouse_env():
    env = WarehouseEnv()
    state, info = env.reset()
    assert len(state) == 2, "State should be (x, y)"
    assert env.robot_pos == [0, 0], "Robot should start at (0, 0)"
    
    # Test step
    next_state, reward, done, truncated, info = env.step(1) # Move down
    assert next_state[1] == 1 or reward == -100, "Robot should move down or hit obstacle"

def test_advanced_env():
    env = AdvancedWarehouseEnv(grid_size=10, num_obstacles=5, num_goals=3, max_battery=100)
    state, info = env.reset()
    assert env.battery == 100, "Battery should be fully charged"
    assert len(env.goals) == 3, "Should have 3 goals"
    
    # Test step
    _, reward, _, _, info = env.step(0)
    assert env.battery == 99, "Battery should decrease by 1"

def test_partial_obs_env():
    env = PartialObsWarehouseEnv()
    state, info = env.reset()
    
    # State should be 5x5 window (25) + dx + dy + battery = 28
    assert len(state) == 28, "Partial obs state should be 28-dimensional"

def test_multi_agent_env():
    env = MultiAgentWarehouseEnv()
    state, info = env.reset()
    
    # State should be 8 dim (r1, r2, g1, g2)
    assert len(state) == 8, "Multi-agent state should be 8-dimensional"
    
    # Test simultaneous step
    next_state, rewards, done, _, _ = env.step((0, 1)) # r1 up, r2 down
    assert len(rewards) == 2, "Should return 2 rewards"

if __name__ == "__main__":
    print("Running basic assertions...")
    test_warehouse_env()
    test_advanced_env()
    test_partial_obs_env()
    test_multi_agent_env()
    print("All environment tests passed successfully!")
