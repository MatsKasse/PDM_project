import math
import random

from my_obstacles import *
from RRT_star.RRT_STAR_definitions import is_line_collision_free, is_point_in_obstacle

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRT:
    def __init__(self, start, goal, obstacles, rand_area, robot_radius=0.3, 
                 expand_dis=0.05, goal_sample_rate=5, max_iter=20000):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []

    def planning(self, pruning=True):
        """
        Main RRT planning loop.
        :param pruning: If True, applies greedy pruning to the final path.
        :return: Final path list [[x,y], ...] (Goal -> Start)
        """
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
            new_node.parent = nearest_node

            if is_point_in_obstacle(new_node.x, new_node.y, self.obstacles, self.robot_radius):
                continue

            # 4. Collision Check & Connect
            if is_line_collision_free(nearest_node.x, nearest_node.y, new_node.x, new_node.y, self.obstacles, self.robot_radius):
                self.node_list.append(new_node)
                
                # Check if we reached the goal region
                if math.hypot(new_node.x - self.goal.x, new_node.y - self.goal.y) <= self.expand_dis:
                    final_node = RRTNode(self.goal.x, self.goal.y)
                    final_node.parent = new_node
                    
                    if is_line_collision_free(new_node.x, new_node.y, final_node.x, final_node.y, self.obstacles, self.robot_radius):
                        raw_path = self.get_path_goal_to_start(final_node)
                        
                        if pruning:
                            return self.simplify_path(raw_path)
                        return raw_path
        
        return None

    def get_path_goal_to_start(self, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append([curr.x, curr.y])
            curr = curr.parent
        return path

    def simplify_path(self, path):
        """
        Greedy Pruning: removes unnecessary waypoints if a straight line exists.
        Expects path in Goal -> Start order (as returned by get_path_goal_to_start).
        """
        if len(path) < 3: return path
        
        # Convert Goal->Start to Start->Goal for logic
        path = path[::-1] 
        simplified = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            for i in range(len(path) - 1, current_idx, -1):
                start_pt = path[current_idx]
                end_pt = path[i]
                
                # Check if we can skip directly to node 'i'
                if is_line_collision_free(start_pt[0], start_pt[1], end_pt[0], end_pt[1], self.obstacles, self.robot_radius):
                    simplified.append(end_pt)
                    current_idx = i
                    break
            else:
                # Fallback step
                current_idx += 1
                simplified.append(path[current_idx])
        
        # Return as Goal -> Start to match RRT convention
        return simplified[::-1]
