import streamlit as st
import os
from PIL import Image

st.set_page_config(page_title="RL Warehouse Dashboard", layout="wide")

st.title("🤖 Autonomous Warehouse Navigation Dashboard")
st.markdown("""
Welcome to the Reinforcement Learning Analytics Dashboard! 
Here you can monitor the training progress and evaluation metrics of the warehouse robots.
""")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Training Curves", "Agent Comparisons", "Path Heatmaps"])

plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')

def display_image(filename, caption):
    filepath = os.path.join(plots_dir, filename)
    if os.path.exists(filepath):
        image = Image.open(filepath)
        # Handle deprecation warning: use_column_width replaced by use_container_width
        st.image(image, caption=caption, use_container_width=True)
    else:
        st.warning(f"Plot '{filename}' not found in the 'plots' folder. Please run the training/evaluation scripts first.")

if page == "Training Curves":
    st.header("📈 Training Performance")
    st.write("Displays the reward convergence over time for individual agents.")
    
    col1, col2 = st.columns(2)
    with col1:
        display_image("rewards.png", "Total Reward per Episode")
    with col2:
        display_image("success.png", "Success Rate over Time")

elif page == "Agent Comparisons":
    st.header("🏆 Agent Benchmarking")
    st.write("Compares the learning efficiency of Q-Learning, DQN, and PPO side-by-side.")
    
    display_image("agent_comparison_rewards.png", "Algorithm Convergence Comparison")

elif page == "Path Heatmaps":
    st.header("🗺️ Spatial Path Heatmaps")
    st.write("Visualizes the exact coordinates the robot visited during evaluation to audit route efficiency.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        display_image("evaluation_heatmap.png", "Evaluation Run Heatmap")
