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
from mpc.mpc_osqp import LinearMPCOSQP, predict_dynamic_obstacles

#part of Lars############################################################

# Global coordinates
sx, sy = 7.5, 7.5   # Start
gx, gy = -8.5, -5   # Goal

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

class RRT:
    def __init__(self, start, goal, obstacles, rand_area, robot_radius=0.4, 
                 expand_dis=1.5, goal_sample_rate=5, max_iter=3000):
        self.start = RRTNode(start[0], start[1])
        self.goal = RRTNode(goal[0], goal[1])
        self.obstacles = obstacles
        self.min_rand, self.max_rand = rand_area
        self.robot_radius = robot_radius
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
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
                        return self.get_path_goal_to_start(final_node)
        
        return None

    def get_path_goal_to_start(self, end_node):
        path = []
        curr = end_node
        while curr is not None:
            path.append([curr.x, curr.y])
            curr = curr.parent
        return path 

def convert_env_obstacles(wall_obs, cyl_obs, dyn_obs):
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
        
    # 2. Cylinders & Dynamic Spheres (Both treated as circles)
    for item in cyl_obs + dyn_obs:
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

#Main function############################################################

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
    
    env: UrdfEnv = UrdfEnv(dt=0.08, robots=robots, render=render, observation_checking=False)
    ob, info = env.reset(pos=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))

    # --- ADD OBSTACLES TO SIMULATION ---
    for wall in wall_obstacles: env.add_obstacle(wall)
    for cylinder in cylinder_obstacles: env.add_obstacle(cylinder)
    for dyn_obst in dynamic_sphere_obstacles: env.add_obstacle(dyn_obst)

    # --------------------------------------------------------
    # Standard RRT Planning Phase
    # --------------------------------------------------------
    print("\n Generating Standard RRT Path (0.4m Radius)...")
    
    rrt_obstacles = convert_env_obstacles(wall_obstacles, cylinder_obstacles, dynamic_sphere_obstacles)
    
    rrt = RRT(start=[sx, sy], goal=[gx, gy], obstacles=rrt_obstacles, 
              rand_area=[-11, 11], robot_radius=0.4, max_iter=3000)
    
    raw_path_list = rrt.planning() # Returns Goal -> Start

    if raw_path_list is None:
        print("Standard RRT Failed to find a path.")
        env.close()
        return []

    # Spline Smoothing
    rx = [p[0] for p in raw_path_list]
    ry = [p[1] for p in raw_path_list]
    
    try:
        if len(rx) < 4:
            rx_smooth, ry_smooth = rx, ry
        else:
            rx_rev, ry_rev = rx[::-1], ry[::-1]
            tck, _ = splprep([rx_rev, ry_rev], s=0.2, k=3)
            u_fine = np.linspace(0, 1, len(rx) * 5)
            rx_s, ry_s = splev(u_fine, tck)
            rx_smooth, ry_smooth = rx_s[::-1], ry_s[::-1]
    except Exception as e:
        print(f"Spline error: {e}. Using raw path.")
        rx_smooth, ry_smooth = rx, ry

    path_xy = np.column_stack((rx_smooth, ry_smooth))

    # Visualization
    plt.figure(figsize=(6,6))
    for obs in rrt_obstacles:
        if obs['type'] == 'rect': 
            plt.gca().add_patch(patches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'], color='gray'))
        elif obs['type'] == 'circle': 
            plt.gca().add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='gray'))
    
    for node in rrt.node_list:
        if node.parent:
            plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g", alpha=0.3, linewidth=0.5)

    plt.plot(rx, ry, "k--", label="Raw RRT")
    plt.plot(rx_smooth, ry_smooth, "r-", linewidth=2, label="Smoothed")
    plt.plot(sx, sy, "go", label="Start")
    plt.plot(gx, gy, "ro", label="Goal")
    plt.legend()
    plt.title("Standard RRT Path Planning")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

#part of Mats ############################################################
    robot_id = 1
    set_robot_body_id(robot_id)
    
    x0 = extract_base_state()
    print(f"\n{'='*60}")
    print(f"Initial state: pos=({x0[0]:.3f}, {x0[1]:.3f}), theta={np.degrees(x0[2]):.1f}°")

    path = path_xy[::-1] # Reverse to Start->Goal
    ref = PolylineReference(path, ds=0.1, v_ref=1.5)   
    
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0) 
    goal_pos = (gx,gy)
    goal_threshold = 0.08  
    
    dx = path[1,0] - path[0,0]
    dy = path[1,1] - path[0,1]
    theta_tangent = np.arctan2(dy, dx)
    
    # MPC setup
    Ts_mpc = 0.08 
    N = 35    
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    Q_matrix = np.diag([25.0, 25.0, 5.0, 5.0])  
    R_matrix = np.diag([0.5, 1.5])          
    P_matrix = np.diag([60.0, 60.0, 15.0, 15.0]) 
    
    mpc = LinearMPCOSQP(
        Ts= Ts_mpc, 
        N= N,
        Q= Q_matrix,  
        R= R_matrix,  
        P= P_matrix,  
        vmin= -0.8,
        vmax= 1.5, 
        wmax= 1.5)

    u_last = np.array([0.0, 0.0])
    
    history = []
    def state_to_sincos(x_state):
        return np.array([x_state[0], x_state[1], np.sin(x_state[2]), np.cos(x_state[2])], dtype=float)

    def min_obs_clearance(pos_xy, obs_pred):
        if not obs_pred: return None
        min_clear = None
        for step in obs_pred:
            for ox, oy, r_safe in step:
                clear = np.hypot(pos_xy[0] - ox, pos_xy[1] - oy) - r_safe
                if min_clear is None or clear < min_clear:
                    min_clear = clear
        return min_clear

    for t in range(n_steps): 
        x = extract_base_state() 
        x_mpc = state_to_sincos(x)

        dist_to_goal = np.linalg.norm([x[0] - goal_pos[0], x[1] - goal_pos[1]])
        if dist_to_goal <= goal_threshold:
            u_last[:] = 0.0
            action = build_action(env.n(), v=0.0, w=0.0)
            ob, reward, terminated, truncated, info = env.step(action)
            history.append((x.copy(), u_last.copy(), ob))
            print(f"\nReached goal within {goal_threshold} m (dist={dist_to_goal:.3f} m). Stopping.")
            break

        if t % steps_per_mpc == 0:
            x_ref, u_ref = ref.horizon(x[0], x[1], x[2], N, use_sincos=True, use_shortest_angle=True, threshold=goal_threshold)
            
            t_attr = getattr(env, "t", None)
            t_now = t_attr() if callable(t_attr) else t * env.dt
            obs_pred = predict_dynamic_obstacles(dynamic_sphere_obstacles, t_now, N, Ts_mpc)  

            u_last, res = mpc.solve(x_mpc, x_ref, u_ref, obs_pred=obs_pred)
            
            status = getattr(res.info, "status", "")
            status_val = getattr(res.info, "status_val", None)
            iters = getattr(res.info, "iter", None)
            pri_res = getattr(res.info, "pri_res", None)
            dua_res = getattr(res.info, "dua_res", None)
            min_clear = min_obs_clearance(x[:2], obs_pred)
            if status_val not in (1, 2) or (t % (steps_per_mpc * 10) == 0):
                print(
                    "[osqp] t={:.2f} status={} iter={} pri_res={} dua_res={} min_clear={} u=({:.3f},{:.3f})".format(
                        t_now,
                        status,
                        iters,
                        None if pri_res is None else round(pri_res, 6),
                        None if dua_res is None else round(dua_res, 6),
                        None if min_clear is None else round(min_clear, 3),
                        u_last[0],
                        u_last[1],
                    )
                )

        action = build_action(env.n(), v=u_last[0], w=u_last[1])
        ob, reward, terminated, truncated, info = env.step(action)
        history.append((x.copy(), u_last.copy(), ob)) 
        
        if terminated or truncated:
            print(f"\n Terminated or Truncated at step {t}")
            break

    x_final = extract_base_state()
    clear_debug_items(path_ids)
    env.close()
    return history

if __name__ == "__main__":
    show_warnings = False
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore" if not show_warnings else "default")
        history = run_albert(n_steps=3000, render=True, path_type="S", path_length=5.0)
