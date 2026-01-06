import warnings
import numpy as np
import pybullet as p
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random

from scipy.interpolate import splprep, splev
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpc.reference_generator import PolylineReference, draw_polyline, clear_debug_items, wrap_angle, create_aligned_path
from mpc.albert_control import extract_base_state, build_action, set_robot_body_id, angle_difference
from mpc.mpc_osqp import LinearMPCOSQP

#Part of Lars ========================================================================

# Global coordinates
sx, sy = 7.5, 7.5   # Start
gx, gy = -6, -8     # Goal

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
                # --- RRT* Logic Starts Here ---
                near_inds = self.find_near_nodes(new_node)
                
                # 4. Choose Best Parent
                new_node = self.choose_parent(new_node, near_inds)
                
                if new_node.parent:
                    self.node_list.append(new_node)
                    # 5. Rewire
                    self.rewire(new_node, near_inds)
        
        # End of iterations: Find best path to goal
        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.get_path_goal_to_start(self.node_list[last_index])
        return None

    def choose_parent(self, new_node, near_inds):
        if not near_inds:
            return new_node
        
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
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

    def steer(self, from_node, to_node):
        new_node = RRTNode(from_node.x, from_node.y)
        d_x = to_node.x - from_node.x
        d_y = to_node.y - from_node.y
        dist = math.hypot(d_x, d_y)
        
        if dist > self.expand_dis:
            dist = self.expand_dis
            
        new_node.x += dist * math.cos(math.atan2(d_y, d_x))
        new_node.y += dist * math.sin(math.atan2(d_y, d_x))
        new_node.cost = from_node.cost + dist
        new_node.parent = from_node
        return new_node

    def find_near_nodes(self, new_node):
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.connect_circle_dist)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2 for node in self.node_list]
        return [i for i, d in enumerate(dist_list) if d <= r**2]

    def rewire(self, new_node, near_inds):
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node: continue
            
            # Check collision and cost improvement
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

    def get_path_goal_to_start(self, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append([curr.x, curr.y])
            curr = curr.parent
        # Add goal if not already there (optional, but good for precision)
        if len(path) > 0 and (path[0][0] != self.goal.x or path[0][1] != self.goal.y):
             path.insert(0, [self.goal.x, self.goal.y])
        return path 

def convert_env_obstacles(wall_obs, cyl_obs, box_obs):
    rrt_obs = []
    for item in wall_obs + box_obs:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue
        w = geo.get('length', geo.get('width', 1.0))
        h = geo.get('width', geo.get('height', 1.0))
        rrt_obs.append({'type': 'rect', 'x': geo['position'][0]-w/2, 'y': geo['position'][1]-h/2, 'w': w, 'h': h})
    for item in cyl_obs:
        if hasattr(item, '_content_dict'): geo = item._content_dict['geometry']
        elif hasattr(item, 'content_dict'): geo = item.content_dict['geometry']
        else: continue
        rrt_obs.append({'type': 'circle', 'x': geo['position'][0], 'y': geo['position'][1], 'r': geo['radius']})
    return rrt_obs

#Main function ========================================================================

def run_albert(n_steps=1000, render=False, path_type="straight", path_length=3.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_offset = np.array([sx, sy, 0.15]),
            spawn_rotation= -0.5*np.pi,
            facing_direction='-x',),]
    
    env: UrdfEnv = UrdfEnv(dt=0.05, robots=robots, render=render, observation_checking=False)
    ob, info = env.reset(pos=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))

    # Set up obstacles
    for wall in wall_obstacles: env.add_obstacle(wall)
    for cylinder in cylinder_obstacles: env.add_obstacle(cylinder)
    for box in box_obstacles: env.add_obstacle(box)

#Part of Lars ========================================================================
    print("\n Generating RRT* Path with 0.4m Radius...")
    rrt_obstacles = convert_env_obstacles(wall_obstacles, cylinder_obstacles, box_obstacles)
    
    # Run RRT*
    rrt_star = RRTStar(start=[sx, sy], goal=[gx, gy], obstacles=rrt_obstacles, rand_area=[-11, 11], robot_radius=0.4)
    raw_path_list = rrt_star.planning() # Returns Goal -> Start

    if raw_path_list is None:
        print("RRT* Failed. Exiting.")
        env.close()
        return []

    # Smoothing (Spline)
    rx = [p[0] for p in raw_path_list] # Goal -> Start X
    ry = [p[1] for p in raw_path_list] # Goal -> Start Y
    
    try:
        # Simple linear interpolation if path is short
        if len(rx) < 4:
            rx_smooth, ry_smooth = rx, ry
        else:
            # We reverse to Start->Goal for smoothing to work logically, then flip back
            rx_rev, ry_rev = rx[::-1], ry[::-1]
            tck, _ = splprep([rx_rev, ry_rev], s=0.5, k=3)
            u_fine = np.linspace(0, 1, len(rx) * 5)
            rx_s, ry_s = splev(u_fine, tck)
            # Flip back to Goal->Start to match A* output expectation
            rx_smooth, ry_smooth = rx_s[::-1], ry_s[::-1]
    except Exception as e:
        print(f"Spline error: {e}. Using raw path.")
        rx_smooth, ry_smooth = rx, ry

    # Resulting path variable for Mats (Must be shape (N, 2) in Goal->Start order)
    path_xy = np.column_stack((rx_smooth, ry_smooth))

    # Visualization of the plan
    plt.figure(figsize=(6,6))
    for obs in rrt_obstacles:
        if obs['type'] == 'rect': plt.gca().add_patch(patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray'))
        elif obs['type'] == 'circle': plt.gca().add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='gray'))
    
    # Draw Tree edges for RRT*
    for node in rrt_star.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.1, linewidth=0.5)

    plt.plot(rx, ry, "b--", label="RRT* Raw")
    plt.plot(rx_smooth, ry_smooth, "r-", label="Smooth")
    plt.plot(sx, sy, "go", label="Start")
    plt.plot(gx, gy, "ro", label="Goal")
    plt.legend()
    plt.title("RRT* Plan")
    plt.show(block=False)
    plt.pause(1)
    plt.close()

#Part of Mats ========================================================================
    robot_id = 1
    set_robot_body_id(robot_id)
    

    # Get initial state (now correctly returns -90° for -y facing robot)
    x0 = extract_base_state()
    print(f"\n{'='*60}")
    print(f"Initial state: pos=({x0[0]:.3f}, {x0[1]:.3f}), theta={np.degrees(x0[2]):.1f}°")

    # Create path aligned with robot's actual heading
    # path = create_aligned_path(x0[0], x0[1], x0[2], path_type, path_length)
    path = path_xy[::-1]
    ref = PolylineReference(path, ds=0.1, v_ref=1.5)   #ds is the resampling interval. Smaller means dense waypoints, more noice
                                                        # Larger ds means fewer reference updates, smoother. But cutting corners.
                                                        #
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0) # Draw path
    goal_pos = (gx,gy)
    goal_threshold = 0.05  # meters; stop when within this distance of goal
    
    dx = path[1,0] - path[0,0]
    dy = path[1,1] - path[0,1]
    theta_tangent = np.arctan2(dy, dx)
    

    # MPC setup
    Ts_mpc = 0.08 #sample period between control decisions made by the MPC. Increase for slower reaction time, lower computation, faster robot.
                    #Decrease when sharp turns, obstacles, more chances per second to correct errors. But increase horizon
    N = 20    #Horizon, amount of steps it looks forward. (basically N * Ts_mpc = seconds looking forward.)
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    Q_matrix = np.diag([30.0, 30.0, 10.0, 10.0])  # Position and sin/cos tracking
    R_matrix = np.diag([0.5, 5.0])          # Control effort (v, w)
    P_matrix = np.diag([60.0, 60.0, 15.0, 15.0])  # Terminal cost
    
    mpc = LinearMPCOSQP(
        Ts= Ts_mpc, 
        N= N,
        Q= Q_matrix,  
        R= R_matrix,  
        P= P_matrix,  
        vmin= 0.0,
        vmax= 3.5, 
        wmax= 0.8)

    u_last = np.array([0.0, 0.0])
    
    history = []
    def state_to_sincos(x_state):
        return np.array([x_state[0], x_state[1], np.sin(x_state[2]), np.cos(x_state[2])], dtype=float)

    for t in range(n_steps): #basically for all t in simulation
        x = extract_base_state() #Extract pose
        x_mpc = state_to_sincos(x)

        dist_to_goal = np.linalg.norm([x[0] - goal_pos[0], x[1] - goal_pos[1]])
        if dist_to_goal <= goal_threshold:
            u_last[:] = 0.0
            action = build_action(env.n(), v=0.0, w=0.0)
            ob, reward, terminated, truncated, info = env.step(action)
            history.append((x.copy(), u_last.copy(), ob))
            print(f"\nReached goal within {goal_threshold} m (dist={dist_to_goal:.3f} m). Stopping.")
            break

        # Update MPC at specified rate
        if t % steps_per_mpc == 0:
            x_ref, u_ref = ref.horizon(x[0], x[1], x[2], N, use_sincos=True, use_shortest_angle=True, threshold=goal_threshold)
            u_last, res = mpc.solve(x_mpc, x_ref, u_ref)
            
            # Print debug info every second
            if t % ( steps_per_mpc) == 0:
                theta_ref = np.arctan2(x_ref[0,2], x_ref[0,3])
                theta_err = np.arctan2(np.sin(theta_ref - x[2]), np.cos(theta_ref - x[2]))
                print("u_last:", u_last, "theta:", x[2], "theta_ref:", theta_ref, "theta_err:", theta_err)
                print("x[0:2]:", x[0:2], "x_ref[0,0:2]:", x_ref[0,0:2], "x_ref[1,0:2]:", x_ref[1,0:2])
                print("osqp:", res.info.status, "iter:", res.info.iter)

        # Apply control
        action = build_action(env.n(), v=u_last[0], w=u_last[1])
        ob, reward, terminated, truncated, info = env.step(action)
        
        history.append((x.copy(), u_last.copy(), ob))  #useful information: True state, controll inputs, observations from simulation.
        
        if terminated or truncated:
            print(f"\n Terminated or Truncated at step {t}, see if collided or reached goal")
            print(info)
            break

    # Final results
    x_final = extract_base_state()
    dist_to_goal = np.linalg.norm([x_final[0] - goal_pos[0], x_final[1] - goal_pos[1]])
    
    clear_debug_items(path_ids)
    env.close()
    return history

if __name__ == "__main__":
    show_warnings = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore" if not show_warnings else "default")
        
        history = run_albert(
            n_steps=3000, 
            render=True,
            path_type="S",  # Options: "straight", "L", "S"
            path_length=5.0
        )
