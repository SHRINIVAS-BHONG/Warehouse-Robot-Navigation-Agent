import sys
import os

def print_menu():
    print("\n" + "="*60)
    print(" 🤖 Warehouse Robot Navigation - Main Menu")
    print("="*60)
    print("\n Training Options:")
    print("  1. Train Q-Learning Agent (Fast, 96.8% success)")
    print("  2. Train Deep Q-Network (DQN)")
    print("  3. Train Proximal Policy Optimization (PPO)")
    print("  4. Train CNN-DQN (Partial Observability)")
    print("  5. Train Multi-Agent System")
    print("  6. Train with Curriculum Learning")
    print("\n Evaluation & Visualization:")
    print("  7. Evaluate Q-Learning Agent (with heatmap)")
    print("  8. Compare All Agents")
    print("  9. Generate All Plots")
    print("\n Other:")
    print("  10. Run Random Agent Demo")
    print("  0. Exit")
    print("="*60)

def main():
    # Ensure we are in the project root
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    if sys.platform == "win32":
        python_exec = os.path.join("venv", "Scripts", "python.exe")
    else:
        python_exec = os.path.join("venv", "bin", "python")
    
    if not os.path.exists(python_exec):
        print(f"Warning: Virtual environment python not found at {python_exec}.")
        print("Falling back to global 'python' command.")
        python_exec = "python"
        
    while True:
        print_menu()
        choice = input("\nSelect an option (0-10): ").strip()
        
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
            script = "generate_plots.py"
        elif choice == '10':
            script = "main.py"
        elif choice == '0':
            print("\n👋 Exiting... Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please select a number between 0-10.")
            input("Press Enter to continue...")
            continue
            
        if script:
            print(f"\n{'='*60}")
            print(f">>> Executing: {script}")
            print(f"{'='*60}\n")
            os.system(f"{python_exec} {script}")
            print(f"\n{'='*60}")
            print("✅ Execution completed!")
            print(f"{'='*60}")
            input("\nPress Enter to return to menu...")

if __name__ == "__main__":
    main()
