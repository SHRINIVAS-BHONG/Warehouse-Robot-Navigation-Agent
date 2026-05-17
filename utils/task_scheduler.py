import itertools

class TaskScheduler:
    """Phase 8 (Step 34): Warehouse Task Scheduling.
    Implements Route Optimization to figure out the most efficient 
    order for the robot to collect all packages (goals).
    Uses brute-force TSP solver (suitable for small number of goals).
    """
    @staticmethod
    def get_optimal_pickup_order(robot_pos, goals):
        """
        Takes current robot position and a list of goal coordinates.
        Returns the optimally ordered list of goals to minimize travel distance.
        """
        if not goals:
            return []
            
        best_order = None
        min_dist = float('inf')
        
        # Check all possible permutations of pickup orders
        for order in itertools.permutations(goals):
            dist = 0
            curr_pos = robot_pos
            for g in order:
                # Manhattan distance is appropriate for grid-based worlds
                dist += abs(curr_pos[0] - g[0]) + abs(curr_pos[1] - g[1])
                curr_pos = g
                
            if dist < min_dist:
                min_dist = dist
                best_order = list(order)
                
        return best_order
