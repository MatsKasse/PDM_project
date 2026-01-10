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
# PART 1: OBSTACLE DEFINITIONS
# ==========================================
def get_obstacles():
    """Imports obstacles directly from my_obstacles.py."""
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
# PART 3: RRT* CLASS
# ==========================================
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, rand_area, 
                 robot_radius=0.3, 
                 expand_dis=1.0, goal_sample_rate=10, max_iter=1000, connect_circle_dist=5.0):
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
                rnd_node = Node(
                    random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand)
                )

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius):
                continue
            
            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_path(last_index)
        else:
            return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist = math.hypot(d_x, d_y)

        if extend_length > dist:
            extend_length = dist

        new_node.x += extend_length * math.cos(math.atan2(d_y, d_x))
        new_node.y += extend_length * math.sin(math.atan2(d_y, d_x))
        new_node.cost = from_node.cost + extend_length
        new_node.parent = from_node
        return new_node

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        return dlist.index(min(dlist))
        
    def find_near_nodes(self, new_node):
        # --- MODIFIED: Uses fixed radius (connect_circle_dist) instead of shrinking ---
        r = self.connect_circle_dist
        
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r**2]
        return near_inds

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
            
            no_collision = is_line_collision_free(new_node.x, new_node.y, near_node.x, near_node.y, self.obstacles, self.robot_radius)
            improved_cost = edge_node.cost < near_node.cost

            if no_collision and improved_cost:
                near_node.parent = new_node
                near_node.cost = edge_node.cost

    def search_best_goal_node(self):
        dist_to_goal_list = [math.hypot(n.x - self.goal.x, n.y - self.goal.y) for n in self.node_list]
        # Use expand_dis as the threshold for RRT* (standard implementation)
        goal_inds = [i for i, d in enumerate(dist_to_goal_list) if d <= self.expand_dis]

        if not goal_inds: return None

        min_cost = min([self.node_list[i].cost for i in goal_inds])
        for i in goal_inds:
            if self.node_list[i].cost == min_cost:
                return i
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
# PART 4: EXECUTION & METRICS
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
    sx, sy = 7.5, 7.5
    
    destination_positions = [
        (-8, -9.5), (-5.5, 8), (9, 0), (2.5, -8), (-4, 0),
        (0, 9.8), (4, -9.5), (-4.4, -5), (9.8, -8), (2, -3.5)
    ]
    
    ROBOT_RADIUS = 0.3
    
    # --- CONFIGURATION (Single Test Setting) ---
    EXPAND_DIS = 0.5
    MAX_ITER = 1500
    CONNECT_DIST = 1.5
    
    # Check Validity
    valid_destinations = []
    print("\nValidating Points...")
    if not validate_point(sx, sy, obstacles, ROBOT_RADIUS):
        print("Start point invalid!")
        return

    for dest in destination_positions:
        if validate_point(dest[0], dest[1], obstacles, ROBOT_RADIUS):
            valid_destinations.append(dest)
    print(f"Valid Destinations: {len(valid_destinations)}")

    # Metrics Storage
    path_lengths = []
    computation_times = []
    success_count = 0

    print(f"\nStarting RRT* Batch Run...")
    print(f"Config: Step={EXPAND_DIS}m, Iter={MAX_ITER}")
    print("-" * 65)
    print(f"{'Run':<5} | {'Goal':<15} | {'Status':<10} | {'Time (s)':<10} | {'Length (m)':<10}")
    print("-" * 65)

    # Plot Setup
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    for obs in obstacles:
        if obs['type'] == 'rect': ax.add_patch(patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray', alpha=0.8))
        elif obs['type'] == 'circle': ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='gray', alpha=0.8))
    plt.plot(sx, sy, "xr", markersize=10, label="Start")

    # Run Loop
    for i, (gx, gy) in enumerate(valid_destinations):
        start_time = time.time()
        
        rrt_star = RRTStar(
            start=[sx, sy], goal=[gx, gy], obstacles=obstacles,
            rand_area=[-11, 11], robot_radius=ROBOT_RADIUS,
            expand_dis=EXPAND_DIS,
            goal_sample_rate=10,
            max_iter=MAX_ITER,
            connect_circle_dist=CONNECT_DIST
        )

        path = rrt_star.planning()
        duration = time.time() - start_time
        computation_times.append(duration)
        
        if path is not None:
            length = calculate_path_length(path)
            path_lengths.append(length)
            success_count += 1
            status = "Success"
            
            # Plot Path
            path_arr = np.array(path)
            plt.plot(path_arr[:, 0], path_arr[:, 1], linewidth=1.5, label=f"Goal {i+1}")
            plt.plot(gx, gy, "xb", markersize=8)
        else:
            length = 0.0
            status = "Failed"
            plt.plot(gx, gy, "xk", markersize=8)

        print(f"{i+1:<5} | ({gx:>5.1f}, {gy:>5.1f})   | {status:<10} | {duration:<10.4f} | {length:<10.4f}")

    # Summary
    print("-" * 65)
    if success_count > 0:
        avg_time = sum(computation_times) / len(computation_times)
        avg_len = sum(path_lengths) / len(path_lengths)
        success_rate = (success_count / len(valid_destinations)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Avg Time:     {avg_time:.4f} s")
        print(f"Avg Length:   {avg_len:.4f} m")
    else:
        print("No paths found.")
    
    plt.title(f"RRT* Batch Run (Success: {success_count}/{len(valid_destinations)})")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.show()

if __name__ == '__main__':
    main()
