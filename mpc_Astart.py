import warnings
import numpy as np
import pybullet as p
import A_star.Gridmap as gm
import gymnasium as gym

from scipy.interpolate import splprep, splev


from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpc.reference_generator import PolylineReference, draw_polyline, clear_debug_items, wrap_angle, create_aligned_path
from mpc.albert_control import extract_base_state, build_action, set_robot_body_id, angle_difference
from mpc.mpc_osqp import LinearMPCOSQP, predict_dynamic_obstacles
from mpc.static_obstacle_to_circles import static_obstacles_to_circles


from A_star.a_star import *


#start world coordinates
sx = 5
sy = 5

#goal world coordinates
gx = 0
gy = 0

#Parameters
robot_radius = 0.3 # robot radius in meters
clearance_weight = 0.5 # weight for clearance in A* cost function
resolution = 0.09 # grid resolution in meters


#Convert world coordinates to grid coordinates
def world_to_grid(x, y, x_min, y_min):
    x_g = int((x - x_min) / resolution)
    y_g = int((y - y_min) / resolution)
    return x_g, y_g

#Convert grid coordinates to world coordinates
def grid_to_world(x_g, y_g, x_min, y_min):
    x = (x_g) * resolution + x_min
    y = (y_g) * resolution + y_min
    return x, y

#setup and run albert with A* and MPC
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
            facing_direction='-y',),]
    
    env: UrdfEnv = UrdfEnv(dt=0.08, robots=robots, render=render, observation_checking=False)
    ob, info = env.reset(pos=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))

    

#Part of Bram ========================================================================
    #Calculate path length
    def get_distance(path_x, path_y):
        path = np.column_stack((path_x, path_y))
        distance = 0
        for i in range(len(path_x)-1):
            distance += np.linalg.norm(path[i]-path[i+1])
        return np.round(distance, 3)

    #Show solution path from A*
    def show_solution(grid, inflated_grid, rx, ry, rx_raw, ry_raw):
        start_grid = (sx, sy)
        goal_grid = (gx, gy)
        path_grid = [(rx[i], ry[i]) for i in range(len(rx))]
        path_grid_raw = [(rx_raw[i], ry_raw[i]) for i in range(len(rx_raw))]

        fig, ax = plt.subplots()

        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1, interpolation="nearest", origin='lower', extent=[x_min, x_max, y_min, y_max])
        ax.imshow(inflated_grid, cmap='Blues', vmin=0, vmax=1, interpolation="nearest", origin='lower', extent=[x_min, x_max, y_min, y_max], alpha=0.5)
        

        ax.plot(start_grid[0], start_grid[1], 'go', markersize=10)  # green circle for start
        ax.plot(goal_grid[0], goal_grid[1], 'ro', markersize=10)    # red circle for goal

        px, py = zip(*path_grid)
        ax.plot(px, py, '-r', linewidth=2)
        px_raw, py_raw = zip(*path_grid_raw)
        ax.plot(px_raw, py_raw, '--g', linewidth=1)
        ax.set_title('A* path with length %1.3fm' %distance)
        ax.grid(True)
        plt.show()

    #Smooth path with spline interpolation
    def spline_smooth(rx, ry, smoothing=0.5):
        tck, _ = splprep([rx, ry], s=smoothing)
        u_fine = np.linspace(0, 1, len(rx) * 5)
        x_s, y_s = splev(u_fine, tck)
        return x_s, y_s
    
    #Set up obstacles in the environment
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    # for cylinder in cylinder_obstacles:
    #     env.add_obstacle(cylinder)

    dynamic_obstacle = True

    if dynamic_obstacle is True:
        for dyn_obst in dynamic_sphere_obstacles:
            env.add_obstacle(dyn_obst)
    for box in box_obstacles:
        env.add_obstacle(box)
    
    world_min, world_max = gm.get_world_bounds()
    x_min, y_min, _ = world_min
    x_max, y_max, _ = world_max

    #Generate gridmap and inflated gridmap
    grid, inflated_grid = gm.generate_gridmap(x_min, x_max, y_min, y_max, resolution=resolution, robot_radius=robot_radius)

    #Convert start and goal to grid coordinates
    sx_g, sy_g = world_to_grid(sx, sy, x_min, y_min)    
    gx_g, gy_g = world_to_grid(gx, gy, x_min, y_min)

    

    #Check if start and goal are in free space
    if inflated_grid[sx_g, sy_g] == 1:
        print("Start is inside an obstacle, please change start position.")
    else:
        print("Start is free.")
    if inflated_grid[gx_g, gy_g] == 1:
        print("Goal is inside an obstacle, please change goal position.")
    else:
        print("Goal is free.")
    
    print('coordinate check', sx_g, sy_g, gx_g, gy_g)

    #Initialize A* planner and plan path
    A_star = AStarPlanner(resolution, inflated_grid, x_min, y_min)
    rx_g, ry_g = A_star.planning(sx_g, sy_g, gx_g, gy_g, weight_clearance=clearance_weight) #weight_clearance --> the clearance to objects
    rx_w, ry_w = zip(*[grid_to_world(x, y, x_min, y_min) for x, y in zip(rx_g, ry_g)])
    
    
    #Smooth the path
    rx_w_smooth, ry_w_smooth = spline_smooth(rx_w, ry_w, 0.1)
    path_xy = np.column_stack((rx_w_smooth, ry_w_smooth)) 


    distance = get_distance(rx_w, ry_w)
    print('distance = ',distance)
    show_solution(grid, inflated_grid, rx_w_smooth, ry_w_smooth, rx_w, ry_w)

    static_circles = static_obstacles_to_circles(wall_obstacles, box_obstacles,
                                                    cylinder_obstacles, robot_radius=robot_radius,
                                                    margin=0.2, sample_radius=0.3)
 


 # Part of Mats ===================================================================== 
    robot_id = 1
    set_robot_body_id(robot_id)
    

    # Get initial state (now correctly returns -90° for -y facing robot)
    x0 = extract_base_state()
    print(f"\n{'='*60}")
    print(f"Initial state: pos=({x0[0]:.3f}, {x0[1]:.3f}), theta={np.degrees(x0[2]):.1f}°")

    # Create path aligned with robot's actual heading
    # path = create_aligned_path(x0[0], x0[1], x0[2], path_type, path_length)
    path = path_xy[::-1]
    ref = PolylineReference(path, ds=0.1, v_ref=1.2)   #ds is the resampling interval. Smaller means dense waypoints, more noice
                                                        # Larger ds means fewer reference updates, smoother. But cutting corners.
                                                        #
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0) # Draw path
    goal_pos = (gx,gy)
    goal_threshold = 0.20  # meters; stop when within this distance of goal
    

    dx = path[1,0] - path[0,0]
    dy = path[1,1] - path[0,1]
    theta_tangent = np.arctan2(dy, dx)
    


    # MPC setup
    Ts_mpc = 0.08 #sample period between control decisions made by the MPC. Increase for slower reaction time, lower computation, faster robot.
                    #Decrease when sharp turns, obstacles, more chances per second to correct errors. But increase horizon
    N = 35    #Horizon, amount of steps it looks forward. (basically N * Ts_mpc = seconds looking forward.)
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    Q_matrix = np.diag([25.0, 25.0, 1.0, 1.0])  # Position and sin/cos tracking
    R_matrix = np.diag([20, 20])          # Control effort (v, w)
    P_matrix = np.diag([60.0, 60.0, 15.0, 15.0])  # Terminal cost
    
    mpc = LinearMPCOSQP(
        Ts= Ts_mpc, 
        N= N,
        Q= Q_matrix,  
        R= R_matrix,  
        P= P_matrix,  
        vmin= -0.5,
        vmax= 1.5, 
        wmax= 1.5)

    u_last = np.array([0.0, 0.0])
    
    history = []
    def state_to_sincos(x_state):
        return np.array([x_state[0], x_state[1], np.sin(x_state[2]), np.cos(x_state[2])], dtype=float)

    def min_obs_clearance(pos_xy, obs_pred):
        if not obs_pred:
            return None
        min_clear = None
        for step in obs_pred:
            for ox, oy, r_safe in step:
                clear = np.hypot(pos_xy[0] - ox, pos_xy[1] - oy) - r_safe
                if min_clear is None or clear < min_clear:
                    min_clear = clear
        return min_clear

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
            

            t_attr = getattr(env, "t", None)
            t_now = t_attr() if callable(t_attr) else t * env.dt

            static_pred = [static_circles for _ in range(N+1)]

            if dynamic_obstacle is True:
                dyn_obs_pred = predict_dynamic_obstacles(dynamic_sphere_obstacles, t_now, N, Ts_mpc)
            else:
                dyn_obs_pred = predict_dynamic_obstacles(None, t_now, N, Ts_mpc)
            
            obs_pred = [dyn_obs_pred[k] + static_pred[k] for k in range(N+1)]

            u_last, res = mpc.solve(x_mpc, x_ref, u_ref, obs_pred=obs_pred)
    

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
