import math
import random


from my_obstacles import *


def get_dist_to_rect(x, y, rect):
    """Calculates shortest distance from point to rectangle."""
    dx = abs(x - (rect['x'] + rect['w'] / 2)) - rect['w'] / 2
    dy = abs(y - (rect['y'] + rect['h'] / 2)) - rect['h'] / 2
    return math.hypot(max(dx, 0), max(dy, 0))

def is_point_in_obstacle(x, y, obstacles, robot_radius):
    for obs in obstacles:
        if obs['type'] == 'rect':
            if get_dist_to_rect(x, y, obs) <= robot_radius: return True
        elif obs['type'] == 'circle':
            if (x - obs['x'])**2 + (y - obs['y'])**2 <= (obs['r'] + robot_radius)**2: return True
    return False

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

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, rand_area, robot_radius=0.4, 
                 expand_dis=1.5, goal_sample_rate=10, max_iter=3000, connect_circle_dist=5.0):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
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
            # 1. Sampling
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = RRTNode(self.goal.x, self.goal.y)
            else:
                rnd_node = RRTNode(random.uniform(self.min_rand, self.max_rand), 
                                   random.uniform(self.min_rand, self.max_rand))

            # 2. Nearest
            dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in self.node_list]
            nearest_ind = dlist.index(min(dlist))
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer
            theta = math.atan2(rnd_node.y - nearest_node.y, rnd_node.x - nearest_node.x)
            new_node = RRTNode(nearest_node.x + self.expand_dis * math.cos(theta), 
                               nearest_node.y + self.expand_dis * math.sin(theta))
            new_node.cost = nearest_node.cost + self.expand_dis
            new_node.parent = nearest_node

            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius):
                continue

            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                # 4. Find Neighbors & Choose Parent
                near_inds = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    # 5. Rewire
                    self.rewire(new_node, near_inds)
        
        # Select best path
        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.get_path_goal_to_start(self.node_list[last_index])
        return None

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

    def steer(self, from_node, to_node):
        new_node = RRTNode(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist = math.hypot(d_x, d_y)
        if dist > self.expand_dis: dist = self.expand_dis
        new_node.x += dist * math.cos(math.atan2(d_y, d_x))
        new_node.y += dist * math.sin(math.atan2(d_y, d_x))
        new_node.cost = from_node.cost + dist
        new_node.parent = from_node
        return new_node

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = max(r, self.expand_dis * 1.1) 
        r = min(r, self.connect_circle_dist)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        return [i for i, d in enumerate(dist_list) if d <= r**2]

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
        goal_inds = [i for i, d in enumerate(dist_to_goal_list) if d <= self.expand_dis]
        if not goal_inds: return None
        
        min_cost = float("inf")
        best_index = None
        for i in goal_inds:
            node = self.node_list[i]
            dist_remain = math.hypot(node.x - self.goal.x, node.y - self.goal.y)
            if node.cost + dist_remain < min_cost:
                min_cost = node.cost + dist_remain
                best_index = i
        return best_index

    def get_path_goal_to_start(self, end_node):
        path = []
        if math.hypot(end_node.x - self.goal.x, end_node.y - self.goal.y) > 0.1:
            path.append([self.goal.x, self.goal.y])
        curr = end_node
        while curr is not None:
            path.append([curr.x, curr.y])
            curr = curr.parent
        return path 

def convert_env_obstacles(wall_obs, cyl_obs, box_obs, dyn_obs):
    """
    Converts environment obstacles to RRT format.
    Specifically extracts initial (t=0) position for dynamic obstacles.
    """
    rrt_obs = []
    
    # 1. Walls (BoxObstacles)
    for item in wall_obs:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue
        
        pos = geo['position']
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        rrt_obs.append({'type': 'rect', 'x': pos[0]-w/2, 'y': pos[1]-h/2, 'w': w, 'h': h})
        
    # 2. Boxes (BoxObstacles)
    for item in box_obs or []:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue

        pos = geo.get('position')
        if pos is None:
            continue
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        rrt_obs.append({'type': 'rect', 'x': pos[0]-w/2, 'y': pos[1]-h/2, 'w': w, 'h': h})

    # 3. Cylinders & Dynamic Spheres (Both treated as circles)
    # for item in cyl_obs + (dyn_obs or []):
    for item in cyl_obs:
        if hasattr(item, '_content_dict'): content = item._content_dict
        elif hasattr(item, 'content_dict'): content = item.content_dict
        else: continue
        
        geo = content.get('geometry', {})
        radius = geo.get('radius', 1.0)
        
        pos = None
        
        # Priority 1: Check for explicit static position in dict
        if 'position' in geo:
            pos = geo['position']
            
        # Priority 2: Use object method with t=0 (Initial Position)
        elif hasattr(item, 'position') and callable(item.position):
            try:
                # Try getting position at time 0 explicitly
                pos = item.position(0)
            except TypeError:
                # Fallback: try without argument (if it doesn't accept time)
                try:
                    pos = item.position()
                except:
                    pos = None
        
        if pos is not None:
            rrt_obs.append({'type': 'circle', 'x': pos[0], 'y': pos[1], 'r': radius})
        
    return rrt_obs
