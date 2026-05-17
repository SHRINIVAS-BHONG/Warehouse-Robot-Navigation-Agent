import os
import numpy as np

# Note: This is a simulated ROS 2 script wrapper. 
# It requires 'rclpy' and a ROS 2 environment to run.
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
except ImportError:
    Node = object # Dummy class to prevent syntax errors on machines without ROS

import tensorflow as tf

class WarehouseRobotController(Node):
    """Phase 8 (Step 35): Sim-to-Real Robotics.
    A ROS 2 Node that bridges our trained TensorFlow PPO agent to control a real 
    or physics-simulated robot (like TurtleBot or Clearpath Jackal).
    
    It subscribes to robot odometry (state) and publishes velocity commands (actions).
    """
    def __init__(self):
        super().__init__('warehouse_robot_controller')
        
        # Load the trained PPO actor model from checkpoints
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints', 'ppo_actor.keras')
        
        try:
            self.actor = tf.keras.models.load_model(model_path)
            self.get_logger().info(f"Policy loaded successfully from {model_path}")
        except Exception as e:
            print(f"[Sim2Real] Model load error or ROS missing: {e}")
            self.actor = None
            
        # ROS Publishers and Subscribers
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )
        
    def odom_callback(self, msg):
        if self.actor is None:
            return
            
        # Extract real robot position from hardware odometry sensors
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # In a full deployment, this would be the PartialObservability CNN data (LiDAR/Camera)
        # For simplicity, we assume the simple (x,y) state mapping here.
        state = np.array([x, y], dtype=np.float32)
        state_input = np.expand_dims(state, axis=0)
        
        # Query neural network policy for best action
        logits = self.actor(state_input)
        action_probs = tf.nn.softmax(logits).numpy()[0]
        action = np.argmax(action_probs) # Exploit purely (no epsilon randomness in production)
        
        # Map discrete RL actions (0:UP, 1:DOWN, 2:LEFT, 3:RIGHT) to continuous hardware velocities
        twist = Twist()
        linear_speed = 0.5 # meters per second
        
        if action == 0:   # Move Forward (+X)
            twist.linear.x = linear_speed
        elif action == 1: # Move Backward (-X)
            twist.linear.x = -linear_speed
        elif action == 2: # Strafe Left (+Y)
            twist.linear.y = linear_speed
        elif action == 3: # Strafe Right (-Y)
            twist.linear.y = -linear_speed
            
        # Send physical voltage commands to motors
        self.publisher_.publish(twist)
        self.get_logger().info(f'Sensor Pos: ({x:.2f}, {y:.2f}) -> Neural Command: {action}')

def main(args=None):
    if Node is object:
        print("ROS 2 (rclpy) is not installed on this system. Sim-to-Real script is in dry-run mode.")
        return
        
    rclpy.init(args=args)
    controller = WarehouseRobotController()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
        
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
