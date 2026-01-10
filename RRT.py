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
    
    # Process Boxes (length=x, width=y)
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
    
    # Check start point
    if is_point_in_obstacle(x1, y1, obstacles, robot_radius): 
        return False
    
    if dist == 0: 
        return True

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

class RRT:
    def __init__(self, start, goal, obstacles, rand_area, 
                 robot_radius=0.4, # <--- NEW DEFAULT PARAMETER
                 expand_dis=1.0, goal_sample_rate=5, max_iter=500):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius # <--- STORE RADIUS
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self):
        self.node_list = [self.start]
        
        for i in range(self.max_iter):
            # 1. Sample a random node (with goal bias)
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = Node(self.goal.x, self.goal.y)
            else:
                rnd_node = Node(
                    random.uniform(self.min_rand, self.max_rand),
                    random.uniform(self.min_rand, self.max_rand)
                )

            # 2. Find nearest node in tree
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer towards random node
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            # 4. Check collision for the path segment (PASS RADIUS HERE)
            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                self.node_list.append(new_node)
                
                # Check if we are close enough to goal
                dist_to_goal = math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y)
                if dist_to_goal <= self.expand_dis:
                    final_node = self.steer(new_node, self.goal, dist_to_goal)
                    if is_line_collision_free(new_node.x, new_node.y, final_node.x, final_node.y, self.obstacles, self.robot_radius):
                        return self.generate_final_path(len(self.node_list) - 1)
        
        return None  # Path not found

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = Node(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist = math.hypot(d_x, d_y)

        if extend_length > dist:
            extend_length = dist

        new_node.x += extend_length * math.cos(math.atan2(d_y, d_x))
        new_node.y += extend_length * math.sin(math.atan2(d_y, d_x))
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


def calculate_path_length(path):
    """Calculates the Euclidean length of the path."""
    length = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i+1]
        length += math.hypot(x2 - x1, y2 - y1)
    return length

def main():
    print("Parsing Obstacles...")
    obstacles = get_obstacles()
    
    sx, sy = -9.0, -9.0
    gx, gy = 9.0, 9.0
    
    # --- DEFINE ROBOT RADIUS ---
    ROBOT_RADIUS = 0.4

    rrt = RRT(
        start=[sx, sy],
        goal=[gx, gy],
        obstacles=obstacles,
        rand_area=[-11, 11],
        robot_radius=ROBOT_RADIUS, # <--- Pass Radius Here
        expand_dis=0.05,
        max_iter=25000
    )

    print(f"Running RRT with Robot Radius: {ROBOT_RADIUS}m ...")
    path = rrt.planning()

    length = 0.0
    if path is None:
        print("Failed to find a path!")
    else:
        length = calculate_path_length(path)
        print("-" * 30)
        print(f"Path Found!")
        print(f"Waypoints: {len(path)}")
        print(f"Total Path Length: {length:.4f} meters")
        print("-" * 30)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Draw Obstacles
    for obs in obstacles:
        if obs['type'] == 'rect':
            rect = patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray')
            ax.add_patch(rect)
        elif obs['type'] == 'circle':
            circle = patches.Circle((obs['x'], obs['y']), obs['r'], color='gray')
            ax.add_patch(circle)

    # Draw Tree
    for node in rrt.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.3, linewidth=0.5)

    # Draw Final Path
    if path is not None:
        path = np.array(path)
        plt.plot(path[:, 0], path[:, 1], '-r', linewidth=2.5, label="Final Path")

    plt.plot(sx, sy, "xr", markersize=10, label="Start")
    plt.plot(gx, gy, "xb", markersize=10, label="Goal")

    title_text = f"RRT (Radius: {ROBOT_RADIUS}m, Length: {length:.2f}m)" if path is not None else "RRT (Fail)"
    plt.title(title_text)
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.axis("equal")
    plt.show()

if __name__ == '__main__':
    main()
