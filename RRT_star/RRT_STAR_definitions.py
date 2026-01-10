import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==========================================
# HELPER FUNCTIONS
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
            if (x - obs['x'])**2 + (y - obs['y'])**2 <= (obs['r'] + robot_radius)**2: return True
    return False

def is_line_collision_free(x1, y1, x2, y2, obstacles, robot_radius):
    dist = math.hypot(x2 - x1, y2 - y1)
    # Optimization: Step size relative to robot radius prevents excessive checks
    step_check = robot_radius * 0.5 
    
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
# RRT* CLASS
# ==========================================
class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

class RRTStar:
    def __init__(self, start, goal, obstacles, rand_area, robot_radius=0.4, 
                 expand_dis=0.75, goal_sample_rate=10, max_iter=1500, connect_circle_dist=1.5):
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

    def planning(self, pruning=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # 1. Sampling
            if random.randint(0, 100) <= self.goal_sample_rate:
                rnd_node = RRTNode(self.goal.x, self.goal.y)
            else:
                rnd_node = RRTNode(random.uniform(self.min_rand, self.max_rand), 
                                   random.uniform(self.min_rand, self.max_rand))

            # 2. Nearest
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            # 3. Steer (Standard expansion)
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius):
                continue

            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                # 4. Find Neighbors
                near_inds = self.find_near_nodes(new_node)
                
                # 5. Choose Parent
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    # 6. Rewire
                    self.rewire(new_node, near_inds)
        
        # Select best path
        last_index = self.search_best_goal_node()
        if last_index is not None:
            raw_path = self.get_path_goal_to_start(self.node_list[last_index])
            if pruning:
                return self.simplify_path(raw_path)
            return raw_path
            
        return None

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Calculates a new node extending from 'from_node' towards 'to_node'.
        If extend_length is inf, it goes all the way (used for rewiring).
        If extend_length is set (expand_dis), it caps the distance (used for stepping).
        """
        new_node = RRTNode(from_node.x, from_node.y)
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
        dlist = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        near_inds = [i for i, d in enumerate(dlist) if d <= r**2]
        
        # OPTIMIZATION: Limit to nearest 15 neighbors to prevent exponential slowdown
        if len(near_inds) > 15:
            near_inds.sort(key=lambda i: dlist[i])
            near_inds = near_inds[:15]
            
        return near_inds

    def choose_parent(self, new_node, near_inds):
        if not near_inds: return new_node
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            # Pass 'inf' to check full connection
            t_node = self.steer(near_node, new_node, float("inf"))
            if t_node and is_line_collision_free(near_node.x, near_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                costs.append(t_node.cost)
            else:
                costs.append(float("inf"))
        min_cost = min(costs)
        if min_cost == float("inf"): return new_node
        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node, float("inf"))
        return new_node

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            # Pass 'inf' to check full connection
            edge_node = self.steer(new_node, near_node, float("inf"))
            if not edge_node: continue
            
            # Pre-check cost to avoid expensive collision check
            if edge_node.cost < near_node.cost:
                if is_line_collision_free(new_node.x, new_node.y, near_node.x, near_node.y, self.obstacles, self.robot_radius):
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

    def simplify_path(self, path):
        """Greedy Pruning"""
        if len(path) < 3: return path
        path = path[::-1] 
        simplified = [path[0]]
        current_idx = 0
        while current_idx < len(path) - 1:
            for i in range(len(path) - 1, current_idx, -1):
                start_pt = path[current_idx]
                end_pt = path[i]
                if is_line_collision_free(start_pt[0], start_pt[1], end_pt[0], end_pt[1], self.obstacles, self.robot_radius):
                    simplified.append(end_pt)
                    current_idx = i
                    break
            else:
                current_idx += 1
                simplified.append(path[current_idx])
        return simplified[::-1]

# ==========================================
# OBSTACLE CONVERSION
# ==========================================
def convert_env_obstacles(wall_obs, cyl_obs, box_obs, dyn_obs):
    """Converts obstacles to RRT format."""
    rrt_obs = []
    
    for item in wall_obs:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue
        pos = geo['position']
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        rrt_obs.append({'type': 'rect', 'x': pos[0]-w/2, 'y': pos[1]-h/2, 'w': w, 'h': h})
        
    for item in box_obs or []:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue
        pos = geo.get('position')
        if pos is None: continue
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        rrt_obs.append({'type': 'rect', 'x': pos[0]-w/2, 'y': pos[1]-h/2, 'w': w, 'h': h})

    for item in cyl_obs:
        if hasattr(item, '_content_dict'): content = item._content_dict
        elif hasattr(item, 'content_dict'): content = item.content_dict
        else: continue
        geo = content.get('geometry', {})
        radius = geo.get('radius', 1.0)
        pos = None
        if 'position' in geo: pos = geo['position']
        elif hasattr(item, 'position') and callable(item.position):
            try: pos = item.position(0)
            except: 
                try: pos = item.position()
                except: pos = None
        if pos is not None:
            rrt_obs.append({'type': 'circle', 'x': pos[0], 'y': pos[1], 'r': radius})
            
    return rrt_obs
