import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random


def get_obstacles():
    
    wall_length = 20
    wall_thickness = 0.1

    wall_obstacles_dicts = [
        {'type': 'box', 'geometry': {'position': [wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': wall_thickness}},
        {'type': 'box', 'geometry': {'position': [0.0, wall_length/2.0, 0.4], 'width': wall_thickness, 'height': 0.8, 'length': wall_length}},
        {'type': 'box', 'geometry': {'position': [0.0, -wall_length/2.0, 0.4], 'width': wall_thickness, 'height': 0.8, 'length': wall_length}},
        {'type': 'box', 'geometry': {'position': [-wall_length/2.0, 0.0, 0.4], 'width': wall_length, 'height': 0.8, 'length': wall_thickness}},
        {'type': 'box', 'geometry': {'position': [6.5, 4.0, 0.4], 'width': wall_thickness, 'height': 0.8, 'length': 7.0}},
        {'type': 'box', 'geometry': {'position': [2.0, 8.0, 0.4], 'width': 4.0, 'height': 0.8, 'length': wall_thickness}},
    ]

    cylinder_obstacles_dicts = [
        {"type": "cylinder", "geometry": {"position": [8.0, -8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [4.0, -8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [0.0, -8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-4.0, -8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-8.0, -8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [6.0, -5.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [2.0, -5.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-2.0, -5.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-6.0, -5.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [0.0, 8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-4.0, 8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-8.0, 8.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-2.0, 5.0, 0.0], "radius": 1.0}},
        {"type": "cylinder", "geometry": {"position": [-6.0, 5.0, 0.0], "radius": 1.0}},
    ]

    box_obstacles_dicts = [
        {'type': 'box', 'geometry': {'position': [7.0, 0.0, 0.0], 'width': 4.0, 'height': 1.0, 'length': 3.0}},
        {'type': 'box', 'geometry': {'position': [2.5, 0.0, 0.0], 'width': 4.0, 'height': 1.0, 'length': 2.0}},
        {'type': 'box', 'geometry': {'position': [-2.5, 0.0, 0.0], 'width': 4.0, 'height': 1.0, 'length': 2.0}},
        {'type': 'box', 'geometry': {'position': [-7.0, 0.0, 0.0], 'width': 4.0, 'height': 1.0, 'length': 3.0}},
    ]

    obstacles = []
    
    # Process Boxes
    all_boxes = wall_obstacles_dicts + box_obstacles_dicts
    for item in all_boxes:
        geo = item['geometry']
        cx, cy = geo['position'][0], geo['position'][1]
        w, h = geo['length'], geo['width'] 
        obstacles.append({'type': 'rect', 'x': cx - w/2, 'y': cy - h/2, 'w': w, 'h': h})

    # Process Cylinders
    for item in cylinder_obstacles_dicts:
        geo = item['geometry']
        obstacles.append({'type': 'circle', 'x': geo['position'][0], 'y': geo['position'][1], 'r': geo['radius']})

    return obstacles

# ==========================================
# PART 2: COLLISION LOGIC (UPDATED FOR RADIUS)
# ==========================================

def get_dist_to_rect(x, y, rect):
    
    # Calculate the distance from the center of the rectangle to the point
    dx = abs(x - (rect['x'] + rect['w'] / 2)) - rect['w'] / 2
    dy = abs(y - (rect['y'] + rect['h'] / 2)) - rect['h'] / 2

    # If dx and dy are negative, the point is inside.
    # We take max(0, d) to ignore the 'inside' portion for the external distance calculation.
    return math.hypot(max(dx, 0), max(dy, 0))

def is_point_in_obstacle(x, y, obstacles, robot_radius):
    
    for obs in obstacles:
        if obs['type'] == 'rect':
            # Exact distance check: collision if distance to rect < robot_radius
            dist = get_dist_to_rect(x, y, obs)
            if dist <= robot_radius:
                return True
        elif obs['type'] == 'circle':
            # Circle check: distance between centers <= sum of radii
            dist_sq = (x - obs['x'])**2 + (y - obs['y'])**2
            if dist_sq <= (obs['r'] + robot_radius)**2:
                return True
    return False

def is_line_collision_free(x1, y1, x2, y2, obstacles, robot_radius, step_check=0.1):
    
    dist = math.hypot(x2 - x1, y2 - y1)
    
    # Check start point collision
    if is_point_in_obstacle(x1, y1, obstacles, robot_radius): return False
    
    if dist == 0: return True

    steps = int(dist / step_check) + 1
    for i in range(steps + 1):
        t = i / steps
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if is_point_in_obstacle(x, y, obstacles, robot_radius):
            return False
    return True


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, rand_area, 
                 robot_radius=0.4, # <--- NEW DEFAULT PARAMETER
                 expand_dis=1.0, goal_sample_rate=10, max_iter=1000, connect_circle_dist=5.0):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius # <--- STORE RADIUS
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.connect_circle_dist = connect_circle_dist 
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]

        for i in range(self.max_iter):
            # Random Sampling
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = Node(
                    random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand)
                )

            # Nearest Node
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            
            # Steer
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # Check Point Collision (PASS RADIUS)
            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius):
                continue
            
            # Check Edge Collision (PASS RADIUS)
            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                
                # Near nodes
                near_inds = self.find_near_nodes(new_node)
                
                # Choose best parent
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    # Rewire
                    self.rewire(new_node, near_inds)

            if i % 200 == 0:
                print(f"Iter: {i} / {self.max_iter}, Nodes: {len(self.node_list)}")

        # Finish
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
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.connect_circle_dist) 
        
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dist_list) if d <= r**2]
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            # Collision check with radius
            if t_node and is_line_collision_free(near_node.x, near_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                costs.append(t_node.cost)
            else:
                costs.append(float("inf"))

        min_cost = min(costs)
        if min_cost == float("inf"):
            return new_node 

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            
            # Collision check with radius
            no_collision = is_line_collision_free(new_node.x, new_node.y, near_node.x, near_node.y, self.obstacles, self.robot_radius)
            improved_cost = edge_node.cost < near_node.cost

            if no_collision and improved_cost:
                near_node.parent = new_node
                near_node.cost = edge_node.cost

    def search_best_goal_node(self):
        dist_to_goal_list = [math.hypot(n.x - self.goal.x, n.y - self.goal.y) for n in self.node_list]
        goal_inds = [i for i, d in enumerate(dist_to_goal_list) if d <= self.expand_dis]

        if not goal_inds:
            return None

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


def calculate_path_length(path):
    """Calculates the Euclidean length of the path."""
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

if __name__ == '__main__':
    print("Initializing RRT* Planner...")
    obstacles = get_obstacles()
    
    sx, sy = -9.0, -9.0
    gx, gy = 9.0, 9.0

    # --- DEFINE ROBOT RADIUS ---
    ROBOT_RADIUS = 0.4
    
    rrt_star = RRTStar(
        start=[sx, sy],
        goal=[gx, gy],
        obstacles=obstacles,
        rand_area=[-11, 11],
        robot_radius=ROBOT_RADIUS, # <--- Pass Radius Here
        expand_dis=1.0,
        goal_sample_rate=10,
        max_iter=4000,
        connect_circle_dist=3.0
    )

    print(f"Running RRT* with Robot Radius: {ROBOT_RADIUS}m ...")
    path = rrt_star.planning()

    length = 0.0
    if path is None:
        print("No path found.")
    else:
        # Calculate length
        length = calculate_path_length(path)
        print("-" * 30)
        print(f"Path Found!")
        print(f"Waypoints: {len(path)}")
        print(f"Total Path Length: {length:.4f} meters")
        print("-" * 30)

    # Plotting
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Draw Obstacles
    for obs in obstacles:
        if obs['type'] == 'rect':
            rect = patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray', alpha=0.8)
            ax.add_patch(rect)
        elif obs['type'] == 'circle':
            circle = patches.Circle((obs['x'], obs['y']), obs['r'], color='gray', alpha=0.8)
            ax.add_patch(circle)

    # Draw Tree
    for node in rrt_star.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.2, linewidth=0.5)

    # Draw Final Path
    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], '-r', linewidth=3, label="Optimal Path")

    plt.plot(sx, sy, "xr", markersize=10, label="Start")
    plt.plot(gx, gy, "xb", markersize=10, label="Goal")
    
    plt.title(f"RRT* (Radius: {ROBOT_RADIUS}m, Length: {length:.2f} m)" if path is not None else "RRT* Planning (Fail)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.show()
