import numpy as np
import math
import random
import time
import csv
import os

# Import obstacle lists from your external file
from my_obstacles import wall_obstacles as wall_obstacles_dicts
from my_obstacles import cylinder_obstacles as cylinder_obstacles_dicts
from my_obstacles import box_obstacles as box_obstacles_dicts

# ==========================================
# PART 1: OBSTACLE DEFINITIONS
# ==========================================
def get_obstacles():
    """Imports obstacles directly without inflation."""
    obstacles = []
    
    all_boxes = wall_obstacles_dicts + box_obstacles_dicts
    
    for item in all_boxes:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue

        cx, cy = geo['position'][0], geo['position'][1]
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        obstacles.append({'type': 'rect', 'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h})

    for item in cylinder_obstacles_dicts:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue
        obstacles.append({'type': 'circle', 'x': geo['position'][0], 'y': geo['position'][1], 'r': geo['radius']})

    return obstacles

# ==========================================
# PART 2: COLLISION LOGIC
# ==========================================
def get_dist_to_rect(x, y, rect):
    dx = abs(x - (rect['x'] + rect['w'] / 2)) - rect['w'] / 2
    dy = abs(y - (rect['y'] + rect['h'] / 2)) - rect['h'] / 2
    return math.hypot(max(dx, 0), max(dy, 0))

def is_point_in_obstacle(x, y, obstacles, robot_radius):
    for obs in obstacles:
        if obs['type'] == 'rect':
            if get_dist_to_rect(x, y, obs) <= robot_radius: return True
        elif obs['type'] == 'circle':
            dist_sq = (x - obs['x'])**2 + (y - obs['y'])**2
            if dist_sq <= (obs['r'] + robot_radius)**2: return True
    return False

def validate_point(x, y, obstacles, robot_radius):
    for obs in obstacles:
        if obs['type'] == 'rect':
            if get_dist_to_rect(x, y, obs) <= robot_radius: return False
        elif obs['type'] == 'circle':
            if math.hypot(x - obs['x'], y - obs['y']) <= obs['r'] + robot_radius: return False
    return True

def is_line_collision_free(x1, y1, x2, y2, obstacles, robot_radius, step_check=0.1):
    dist = math.hypot(x2 - x1, y2 - y1)
    if is_point_in_obstacle(x1, y1, obstacles, robot_radius): return False
    if dist == 0: return True
    steps = int(dist / step_check) + 1
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if is_point_in_obstacle(x, y, obstacles, robot_radius): return False
    return True

# ==========================================
# PART 3: RRT CLASS
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacles, rand_area, 
                 robot_radius=0.4, expand_dis=1.0, goal_threshold=2.0, 
                 goal_sample_rate=5, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius 
        self.expand_dis = expand_dis       
        self.goal_threshold = goal_threshold 
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = Node(random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand))

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                self.node_list.append(new_node)
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.goal_threshold:
                    final_node = self.steer(new_node, self.goal, dist=math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y))
                    if is_line_collision_free(new_node.x, new_node.y, final_node.x, final_node.y, self.obstacles, self.robot_radius):
                        return self.generate_final_path(len(self.node_list) - 1)
        return None

    def steer(self, from_node, to_node, dist=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist_actual = math.hypot(d_x, d_y)
        if dist > dist_actual: dist = dist_actual
        new_node.x += dist * math.cos(math.atan2(d_y, d_x))
        new_node.y += dist * math.sin(math.atan2(d_y, d_x))
        new_node.parent = from_node
        return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def generate_final_path(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

# ==========================================
# PART 4: UTILS & SMOOTHING
# ==========================================
def simplify_path(path, obstacles, robot_radius):
    if len(path) < 3: return path
    path = path[::-1] # Start -> Goal
    simplified = [path[0]]
    current_idx = 0
    while current_idx < len(path) - 1:
        for i in range(len(path) - 1, current_idx, -1):
            start_pt = path[current_idx]
            end_pt = path[i]
            if is_line_collision_free(start_pt[0], start_pt[1], end_pt[0], end_pt[1], obstacles, robot_radius):
                simplified.append(end_pt)
                current_idx = i
                break
        else:
            current_idx += 1
            simplified.append(path[current_idx])
    return simplified[::-1] # Goal -> Start

def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

# ==========================================
# PART 5: AUTOMATED BENCHMARK
# ==========================================
def main():
    obstacles = get_obstacles()
    sx, sy = -9.0, -9.0
    
    destination_positions = [
        (-8, -9.5), (-5.5, 8), (9, 0), (2.5, -8), (-4, 0),
        (0, 9.8), (4, -9.5), (-4.4, -5), (9.8, -8), (2, -3.5)
    ]
    
    ROBOT_RADIUS = 0.4
    
    # 1. Filter Valid Destinations First
    valid_destinations = []
    for dest in destination_positions:
        if validate_point(dest[0], dest[1], obstacles, ROBOT_RADIUS):
            valid_destinations.append(dest)
    print(f"Destinations Validated: {len(valid_destinations)}/{len(destination_positions)} are reachable.")

    # 2. Define Test Range (Step Sizes)
    test_step_sizes = np.array([2.5, 2, 1.5, 1, 0.5, 0.25, 0.1, 0.075, 0.05, 0.025])
    MAX_ITER = 25000
    GOAL_THRESHOLD = 2.0
    NUM_TRIALS = 3  # <--- Run each specific goal 3 times
    
    filename = "rrt_benchmark_results_3trials.csv"
    
    print(f"\nStarting Automation ({NUM_TRIALS} trials per goal)...")
    print(f"Results will be saved to: {filename}")
    print("-" * 70)
    print(f"{'Step Size':<10} | {'Avg Time (s)':<15} | {'Avg Length (m)':<15} | {'Success Rate':<15}")
    print("-" * 70)

    # Initialize CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Step Size", "Avg Computation Time (s)", "Avg Path Length (m)", "Success Rate (%)"])

        # 3. Loop through Step Sizes
        for step_size in test_step_sizes:
            
            batch_times = []
            batch_lengths = []
            total_attempts = 0
            total_successes = 0
            
            # Loop through all destinations for this setting
            for gx, gy in valid_destinations:
                
                # --- INNER LOOP: 3 TRIALS PER DESTINATION ---
                for trial in range(NUM_TRIALS):
                    total_attempts += 1
                    start_time = time.time()
                    
                    rrt = RRT(
                        start=[sx, sy], goal=[gx, gy], obstacles=obstacles,
                        rand_area=[-11, 11], robot_radius=ROBOT_RADIUS,
                        expand_dis=step_size, 
                        goal_threshold=GOAL_THRESHOLD, 
                        goal_sample_rate=5, 
                        max_iter=MAX_ITER
                    )

                    raw_path = rrt.planning()
                    
                    # Pruning
                    final_path = None
                    if raw_path is not None:
                        final_path = simplify_path(raw_path, obstacles, ROBOT_RADIUS)
                    
                    duration = time.time() - start_time
                    batch_times.append(duration)
                    
                    if final_path is not None:
                        length = calculate_path_length(final_path)
                        batch_lengths.append(length)
                        total_successes += 1
            
            # Calculate Averages for this Step Size (over all trials)
            avg_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
            avg_len = sum(batch_lengths) / len(batch_lengths) if batch_lengths else 0.0
            success_rate = (total_successes / total_attempts) * 100
            
            # Print to Console
            print(f"{step_size:<10.4f} | {avg_time:<15.4f} | {avg_len:<15.4f} | {success_rate:<15.1f}")
            
            # Write to CSV
            writer.writerow([step_size, avg_time, avg_len, success_rate])

    print("-" * 70)
    print("Benchmark Completed.")

if __name__ == '__main__':
    main()
