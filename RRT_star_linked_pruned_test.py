import numpy as np
import math
import random
import time
import csv
import os

# Import obstacles
from my_obstacles import wall_obstacles as wall_obstacles_dicts
from my_obstacles import cylinder_obstacles as cylinder_obstacles_dicts
from my_obstacles import box_obstacles as box_obstacles_dicts

# ==========================================
# PART 1: OBSTACLE PARSING
# ==========================================
def get_obstacles():
    obstacles = []
    all_boxes = wall_obstacles_dicts + box_obstacles_dicts
    for item in all_boxes:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue
        cx, cy = geo['position'][0], geo['position'][1]
        w, h = geo.get('length', 1.0), geo.get('width', 1.0)
        obstacles.append({'type': 'rect', 'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h})
    for item in cylinder_obstacles_dicts:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue
        obstacles.append({'type': 'circle', 'x': geo['position'][0], 'y': geo['position'][1], 'r': geo['radius']})
    return obstacles

# ==========================================
# PART 2: COLLISION & UTILS
# ==========================================
def is_point_in_obstacle(x, y, obstacles, robot_radius):
    for obs in obstacles:
        if obs['type'] == 'rect':
            dx = abs(x - (obs['x'] + obs['w'] / 2)) - obs['w'] / 2
            dy = abs(y - (obs['y'] + obs['h'] / 2)) - obs['h'] / 2
            if math.hypot(max(dx, 0), max(dy, 0)) <= robot_radius: return True
        elif obs['type'] == 'circle':
            if (x - obs['x'])**2 + (y - obs['y'])**2 <= (obs['r'] + robot_radius)**2: return True
    return False

def validate_point(x, y, obstacles, robot_radius):
    for obs in obstacles:
        if obs['type'] == 'rect':
            dx = abs(x - (obs['x'] + obs['w'] / 2)) - obs['w'] / 2
            dy = abs(y - (obs['y'] + obs['h'] / 2)) - obs['h'] / 2
            if math.hypot(max(dx, 0), max(dy, 0)) <= robot_radius: return False
        elif obs['type'] == 'circle':
            if (x - obs['x'])**2 + (y - obs['y'])**2 <= (obs['r'] + robot_radius)**2: return False
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

def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

def simplify_path(path, obstacles, robot_radius):
    """Greedy pruning for post-processing."""
    if len(path) < 3: return path
    path = path[::-1] 
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
    return simplified[::-1] 

# ==========================================
# PART 3: RRT* CLASS
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, rand_area, 
                 robot_radius=0.3, expand_dis=0.5, 
                 goal_sample_rate=10, max_iter=1000, connect_circle_dist=1.5):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist 
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = Node(random.uniform(self.min_rand, self.max_rand), 
                                random.uniform(self.min_rand, self.max_rand))

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius): continue
            
            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_path(last_index)
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist = math.hypot(d_x, d_y)
        if extend_length > dist: extend_length = dist
        new_node.x += extend_length * math.cos(math.atan2(d_y, d_x))
        new_node.y += extend_length * math.sin(math.atan2(d_y, d_x))
        new_node.cost = from_node.cost + extend_length
        new_node.parent = from_node
        return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))

    def find_near_nodes(self, new_node):
        r = self.connect_circle_dist
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        return [i for i, d in enumerate(dist_list) if d <= r**2]

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and is_line_collision_free(near_node.x, near_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                costs.append(t_node.cost)
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return new_node
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            if is_line_collision_free(new_node.x, new_node.y, near_node.x, near_node.y, self.obstacles, self.robot_radius):
                if edge_node.cost < near_node.cost:
                    near_node.parent = new_node
                    near_node.cost = edge_node.cost

    def search_best_goal_node(self):
        dist_to_goal_list = [math.hypot(n.x - self.goal.x, n.y - self.goal.y) for n in self.node_list]
        goal_inds = [i for i, d in enumerate(dist_to_goal_list) if d <= self.expand_dis]
        if not goal_inds: return None
        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost: return i
        return None

    def generate_final_path(self, goal_ind):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([self.start.x, self.start.y])
        return path

# ==========================================
# PART 4: AUTOMATION
# ==========================================
def main():
    obstacles = get_obstacles()
    sx, sy = 7.5,7.5
    
    destination_positions = [
        (-8, -9.5), (-5.5, 8), (9, 0), (2.5, -8), (-4, 0),
        (0, 9.8), (4, -9.5), (-4.4, -5), (9.8, -8), (2, -3.5)
    ]
    
    ROBOT_RADIUS = 0.3
    
    # Filter valid destinations
    valid_destinations = []
    for dest in destination_positions:
        if validate_point(dest[0], dest[1], obstacles, ROBOT_RADIUS):
            valid_destinations.append(dest)
    print(f"Validated {len(valid_destinations)} destinations.")

    # --- TEST SETTINGS (LOWER RANGE) ---
    # Range: 100 to 3000 iterations
    test_iterations = [750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]
    
    # --- YOUR SPECIFIC PARAMS ---
    FIXED_STEP = 0.75
    FIXED_CONNECT = 1.5
    NUM_TRIALS = 1
    
    filename = "rrt_star_benchmark_low_iter.csv"
    
    print(f"\nStarting RRT* Benchmark (Low Iterations)...")
    print(f"Params: Step={FIXED_STEP}m, Rewire={FIXED_CONNECT}m")
    print(f"Results will be saved to: {filename}")
    print("-" * 80)
    print(f"{'Max Iter':<10} | {'Avg Time (s)':<15} | {'Avg Length (m)':<15} | {'Success Rate':<15}")
    print("-" * 80)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Max Iterations", "Avg Computation Time (s)", "Avg Path Length (m)", "Success Rate (%)"])

        for max_iter in test_iterations:
            
            batch_times = []
            batch_lengths = []
            total_attempts = 0
            total_successes = 0
            
            for gx, gy in valid_destinations:
                for trial in range(NUM_TRIALS):
                    total_attempts += 1
                    start_time = time.time()
                    
                    rrt_star = RRTStar(
                        start=[sx, sy], goal=[gx, gy], obstacles=obstacles,
                        rand_area=[-11, 11], robot_radius=ROBOT_RADIUS,
                        expand_dis=FIXED_STEP, 
                        max_iter=max_iter,
                        connect_circle_dist=FIXED_CONNECT,
                        goal_sample_rate=10
                    )

                    path = rrt_star.planning()
                    
                    # Apply Pruning 
                    if path is not None:
                        path = simplify_path(path, obstacles, ROBOT_RADIUS)
                    
                    duration = time.time() - start_time
                    batch_times.append(duration)
                    
                    if path is not None:
                        length = calculate_path_length(path)
                        batch_lengths.append(length)
                        total_successes += 1
            
            # Stats
            avg_time = sum(batch_times) / len(batch_times) if batch_times else 0.0
            avg_len = sum(batch_lengths) / len(batch_lengths) if batch_lengths else 0.0
            success_rate = (total_successes / total_attempts) * 100
            
            print(f"{max_iter:<10} | {avg_time:<15.4f} | {avg_len:<15.4f} | {success_rate:<15.1f}")
            writer.writerow([max_iter, avg_time, avg_len, success_rate])

    print("-" * 80)
    print("Benchmark Completed.")

if __name__ == '__main__':
    main()
