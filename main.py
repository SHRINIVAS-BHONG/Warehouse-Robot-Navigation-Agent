from env.warehouse_env import WarehouseEnv
import random

env = WarehouseEnv()

state, info = env.reset()

done = False

while not done:

    action = random.randint(0, 3)

    state, reward, done, truncated, info = env.step(action)

    env.render()

    print(
        "State:", state,
        "Reward:", reward
    )