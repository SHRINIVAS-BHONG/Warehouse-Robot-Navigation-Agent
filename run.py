import sys
import os

def print_menu():
    print("\n" + "="*50)
    print(" Warehouse RL Navigation - Central CLI Runner")
    print("="*50)
    print(" 1. Train Q-Learning Agent")
    print(" 2. Train Deep Q-Network (DQN)")
    print(" 3. Train Proximal Policy Optimization (PPO)")
    print(" 4. Train CNN Vision Agent (Partial Observability)")
    print(" 5. Train Multi-Agent System (Decentralized DQN)")
    print(" 6. Run Curriculum Learning (Auto-Scaling Difficulty)")
    print(" 7. Evaluate Trained Q-Learning Agent (w/ Heatmap)")
    print(" 8. Run Agent Benchmark & Comparison Suite")
    print(" 9. Exit")
    print("="*50)

def main():
    # Ensure we are in the project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    python_exec = os.path.join("venv", "Scripts", "python")
    
    if not os.path.exists(python_exec):
        print("Warning: Virtual environment python not found at venv\\Scripts\\python.")
        print("Falling back to global 'python' command.")
        python_exec = "python"
        
    while True:
        print_menu()
        choice = input("Select a pipeline to execute (1-9): ")
        
        script = None
        if choice == '1':
            script = os.path.join("training", "train_q.py")
        elif choice == '2':
            script = os.path.join("training", "train_dqn.py")
        elif choice == '3':
            script = os.path.join("training", "train_ppo.py")
        elif choice == '4':
            script = os.path.join("training", "train_cnn_dqn.py")
        elif choice == '5':
            script = os.path.join("training", "train_multi_agent.py")
        elif choice == '6':
            script = os.path.join("training", "train_curriculum.py")
        elif choice == '7':
            script = os.path.join("training", "eval_q.py")
        elif choice == '8':
            script = os.path.join("training", "compare_agents.py")
        elif choice == '9':
            print("Exiting CLI...")
            sys.exit(0)
        else:
            print("Invalid choice. Please select a valid number.")
            continue
            
        if script:
            print(f"\n>>> Executing: {python_exec} {script}\n")
            os.system(f"{python_exec} {script}")

if __name__ == "__main__":
    main()
