# Autonomous Warehouse Robot Navigation

This repository contains a professional-grade Reinforcement Learning pipeline designed to train autonomous robotic agents for navigation and task execution inside a warehouse environment. 

The project progressively builds intelligence from basic tabular Q-Learning up to Deep Reinforcement Learning (DQN, PPO) and advanced Computer Vision navigation.

## 🚀 Project Features

*   **Custom Reinforcement Learning Environments:** From static grids to dynamic, partial-observability, and multi-agent warehouse simulations.
*   **Deep Learning Architectures:** Integrated with TensorFlow/Keras to build Dense and CNN-based neural networks.
*   **Curriculum Learning:** Automated progressive difficulty scaling.
*   **Sim-to-Real Pipeline:** Includes ROS 2 nodes for porting trained policies onto physical hardware.
*   **Visual Analytics:** Automatically generates convergence curves, multi-agent comparisons, and spatial heatmaps.

## 📂 Folder Structure

```text
reinforcement_project/
├── agents/                     # Policy Definitions
│   ├── q_learning_agent.py     # Tabular Q-Table
│   ├── dqn_agent.py            # Deep Q-Network
│   ├── cnn_dqn_agent.py        # CNN Vision DQN
│   └── ppo_agent.py            # Proximal Policy Optimization
├── env/                        # Environments
│   ├── warehouse_env.py        # Basic static environment
│   ├── advanced_warehouse_env.py # Dynamic obstacles, multiple goals, battery
│   ├── partial_obs_env.py      # 5x5 sensor window
│   └── multi_agent_env.py      # Dual robot navigation
├── training/                   # Execution Pipelines
│   ├── train_q.py              # Train tabular Q-Learning
│   ├── train_dqn.py            # Train standard DQN
│   ├── train_ppo.py            # Train PPO Actor-Critic
│   ├── train_cnn_dqn.py        # Train Vision Navigation
│   ├── train_curriculum.py     # Execute Curriculum Learning
│   ├── compare_agents.py       # Benchmark all agents
│   └── eval_q.py               # Run and visualize trained models
├── utils/                      # Helper Modules
│   ├── replay_buffer.py        # Experience replay for DQN
│   ├── visualization.py        # Matplotlib analytics
│   └── task_scheduler.py       # TSP optimal route planner
├── sim2real/                   # Robotics Middleware
│   └── ros_node.py             # ROS 2 hardware bridge
├── checkpoints/                # Saved TensorFlow models & Pickles
└── plots/                      # Generated visualization graphs
```

## 🛠️ Installation

1. **Activate your Virtual Environment** (Recommended):
   Ensure your Python virtual environment is activated before installing dependencies.
   ```powershell
   venv\Scripts\activate
   ```

2. **Install Dependencies**:
   The primary dependencies for this repository are `numpy`, `tensorflow`, `matplotlib`, and `seaborn`.
   ```powershell
   pip install numpy tensorflow matplotlib seaborn
   ```

## 🧠 Training Instructions

You can orchestrate training for any algorithm by running the respective script in the `training` directory. All models will automatically save to the `checkpoints/` folder.

**Tabular Q-Learning:**
```powershell
python training/train_q.py
```

**Deep Q-Network (DQN):**
```powershell
python training/train_dqn.py
```

**Proximal Policy Optimization (PPO):**
```powershell
python training/train_ppo.py
```

**CNN Vision Navigation:**
```powershell
python training/train_cnn_dqn.py
```

**Curriculum Learning (Auto-Scaling):**
```powershell
python training/train_curriculum.py
```

## 📊 Analytics & Benchmarking

To test which algorithm performs best in the environment, run the comparison suite. This will run short training loops and plot their reward curves side-by-side in the `plots/` directory:

```powershell
python training/compare_agents.py
```

## 🤖 Sim-to-Real Robotics

If you have ROS 2 installed on your machine or inside a container, you can spin up the hardware-integration node which automatically subscribes to `/odom` and publishes `/cmd_vel` using the trained PPO neural network.

```powershell
python sim2real/ros_node.py
```

## 🏆 Resume Value
This project demonstrates expertise in:
- Reinforcement Learning Engineering
- Deep Learning & TensorFlow Architecture
- Environment Simulation Design
- Autonomous Systems & Computer Vision Navigation
- Professional Software Architecture
