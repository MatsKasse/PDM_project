import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random
import time

# Import obstacle lists from your external file
from my_obstacles import wall_obstacles as wall_obstacles_dicts
from my_obstacles import cylinder_obstacles as cylinder_obstacles_dicts
from my_obstacles import box_obstacles as box_obstacles_dicts

# ==========================================
# PART 1: OBSTACLE DEFINITIONS (RAW IMPORT)
# ==========================================
def get_obstacles():
    """
    Imports obstacles directly from my_obstacles.py without inflation.
    """
    obstacles = []
    
    # Process Wall & Box Objects (treated as rectangles)
    all_boxes = wall_obstacles_dicts + box_obstacles_dicts
    
    for item in all_boxes:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue

        cx, cy = geo['position'][0], geo['position'][1]
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        
        # Base rectangle coords (top-left)
        x = cx - w/2
        y = cy - h/2
        
        obstacles.append({'type': 'rect', 'x': x, 'y': y, 'w': w, 'h': h})

    # Process Cylinders
    for item in cylinder_obstacles_dicts:
        if isinstance(item, dict): geo = item['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        elif hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        else: continue
             
        obstacles.append({
            'type': 'circle', 
            'x': geo['position'][0], 
            'y': geo['position'][1], 
            'r': geo['radius']
        })

    return obstacles

# ==========================================
# PART 2: COLLISION LOGIC (WITH ROBOT RADIUS)
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

def validate_point(x, y, obstacles, robot_radius, name="Point"):
    for i, obs in enumerate(obstacles):
        collision = False
        if obs['type'] == 'rect':
            if get_dist_to_rect(x, y, obs) <= robot_radius: collision = True
        elif obs['type'] == 'circle':
            if math.hypot(x - obs['x'], y - obs['y']) <= obs['r'] + robot_radius: collision = True
        
        if collision:
            return False
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
                 robot_radius=0.4, expand_dis=0.5, goal_threshold=2.0, 
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
# PART 4: PATH SMOOTHING (PRUNING)
# ==========================================
def simplify_path(path, obstacles, robot_radius):
    if len(path) < 3: return path
    
    path = path[::-1] # Convert to Start -> Goal
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
            
    return simplified[::-1] # Return as Goal -> Start

# ==========================================
# PART 5: EXECUTION
# ==========================================
def calculate_path_length(path):
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

def main():
    print("Parsing Obstacles...")
    obstacles = get_obstacles()
    sx, sy = 7.5,7.5
    
    destination_positions = [
        (-8, -9.5), (-5.5, 8), (9, 0), (2.5, -8), (-4, 0),
        (0, 9.8), (4, -9.5), (-4.4, -5), (9.8, -8), (2, -3.5)
    ]
    
    # --- CHANGED: Using Real Robot Radius ---
    ROBOT_RADIUS = 0.3 
    STEP_SIZE = 0.025
    GOAL_THRESHOLD = 2.0
    MAX_ITER = 25000
    
    valid_destinations = []
    print("\nValidating Points...")
    for dest in destination_positions:
        if validate_point(dest[0], dest[1], obstacles, ROBOT_RADIUS, f"Goal {dest}"):
            valid_destinations.append(dest)
    
    path_lengths = []
    computation_times = []
    success_count = 0
    
    # Plotting
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # Draw Obstacles
    for obs in obstacles:
        if obs['type'] == 'rect': ax.add_patch(patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray'))
        elif obs['type'] == 'circle': ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='gray'))
    plt.plot(sx, sy, "xr", markersize=10, label="Start")

    print(f"\n{'Run':<5} | {'Goal':<15} | {'Status':<10} | {'Time':<10} | {'Len (Raw)':<10} | {'Len (Smooth)':<12}")
    print("-" * 80)

    for i, (gx, gy) in enumerate(valid_destinations):
        start_time = time.time()
        
        rrt = RRT(
            start=[sx, sy], goal=[gx, gy], obstacles=obstacles,
            rand_area=[-11, 11], robot_radius=ROBOT_RADIUS,
            expand_dis=STEP_SIZE, goal_threshold=GOAL_THRESHOLD, 
            goal_sample_rate=5, max_iter=MAX_ITER
        )

        raw_path = rrt.planning()
        
        # Pruning
        final_path = None
        if raw_path is not None:
            final_path = simplify_path(raw_path, obstacles, ROBOT_RADIUS)
        
        duration = time.time() - start_time
        computation_times.append(duration)
        
        if final_path is not None:
            raw_len = calculate_path_length(raw_path)
            smooth_len = calculate_path_length(final_path)
            path_lengths.append(smooth_len)
            success_count += 1
            
            path_arr = np.array(final_path)
            plt.plot(path_arr[:, 0], path_arr[:, 1], linewidth=2.0, label=f"Path {i+1}")
            plt.plot(gx, gy, "xb") 
            status = "Success"
        else:
            raw_len = 0.0
            smooth_len = 0.0
            status = "Failed"
            plt.plot(gx, gy, "xk") 

        print(f"{i+1:<5} | ({gx:>5.1f}, {gy:>5.1f})   | {status:<10} | {duration:<10.4f} | {raw_len:<10.4f} | {smooth_len:<12.4f}")

    print("-" * 80)
    if success_count > 0:
        print(f"Success Rate: {(success_count/len(valid_destinations))*100:.1f}%")
        print(f"Avg Time: {sum(computation_times)/len(computation_times):.4f}s")
        print(f"Avg Length: {sum(path_lengths)/len(path_lengths):.4f}m")
    
    plt.title("Standard RRT with Pruning")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
