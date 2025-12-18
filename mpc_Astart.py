import warnings
import numpy as np
import pybullet as p
import Gridmap as gm
import gymnasium as gym

from scipy.interpolate import splprep, splev


from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpc.reference_generator import PolylineReference, draw_polyline, clear_debug_items, wrap_angle, create_aligned_path
from mpc.albert_control import extract_base_state, build_action, set_robot_body_id, angle_difference
from mpc.mpc_osqp import LinearMPCOSQP

from a_star import *

x_min = 0
y_min = 0

sx = 0.0
sy = 0.0
gx = -8.0
gy = -5




visits = 3 # amount of random points to visit

resolution = 0.09

def world_to_grid(x, y, x_min, y_min):
    x_g = int((x - x_min) / resolution)
    y_g = int((y - y_min) / resolution)
    return x_g, y_g

def grid_to_world(x_g, y_g, x_min, y_min):
    x = (x_g) * resolution + x_min
    y = (y_g) * resolution + y_min
    return x, y




def run_albert(n_steps=1000, render=False, path_type="straight", path_length=3.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_offset = np.array([0, 0, 0.15]),
            spawn_rotation=0,
            facing_direction='-y',),]
    
    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)
    ob, info = env.reset(pos=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))

    # Add obstacles
    # for wall in wall_obstacles:
    #     env.add_obstacle(wall)
    # for cylinder in cylinder_obstacles:
    #     env.add_obstacle(cylinder)
    # for box in box_obstacles:
    #     env.add_obstacle(box)

#Part of Bram ========================================================================
    def get_distance(path_x, path_y):
        path = np.column_stack((path_x, path_y))
        distance = 0
        for i in range(len(path_x)-1):
            distance += np.linalg.norm(path[i]-path[i+1])
        return np.round(distance, 3)
    
    def show_solution(grid, rx, ry):
        start_grid = (sx, sy)
        goal_grid = (gx, gy)
        path_grid = [(rx[i], ry[i]) for i in range(len(rx))]

        fig, ax = plt.subplots()
    
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1, interpolation="nearest", origin='lower', extent=[x_min, x_max, y_min, y_max])

        ax.plot(start_grid[0], start_grid[1], 'go', markersize=10)  # green circle for start
        ax.plot(goal_grid[0], goal_grid[1], 'ro', markersize=10)    # red circle for goal

        px, py = zip(*path_grid)
        ax.plot(px, py, '-r', linewidth=2)
        ax.set_title('A* path with length %1.3fm' %distance)
        ax.grid(True)
        plt.show()

    for wall in wall_obstacles:
        env.add_obstacle(wall)
    for cylinder in cylinder_obstacles:
        env.add_obstacle(cylinder)
    # for box in box_obstacles:
    #     env.add_obstacle(box)
    
    world_min, world_max = gm.get_world_bounds()
    x_min, y_min, _ = world_min
    x_max, y_max, _ = world_max

    
    grid, inflated_grid = gm.generate_gridmap(x_min, x_max, y_min, y_max, resolution=resolution)

    sx_g, sy_g = world_to_grid(sx, sy, x_min, y_min)    

    gx_g, gy_g = world_to_grid(gx, gy, x_min, y_min)

    print('coordinate check', sx_g, sy_g, gx_g, gy_g)


    A_star = AStarPlanner(resolution, 0.3, inflated_grid, x_min, y_min)
    rx_g, ry_g = A_star.planning(sx_g, sy_g, gx_g, gy_g)
    rx_w, ry_w = zip(*[grid_to_world(x, y, x_min, y_min) for x, y in zip(rx_g, ry_g)])


    def spline_smooth(rx, ry, smoothing=0.5):
        tck, _ = splprep([rx, ry], s=smoothing)
        u_fine = np.linspace(0, 1, len(rx) * 5)
        x_s, y_s = splev(u_fine, tck)
        return x_s, y_s
    
    
    rx_w_smooth, ry_w_smooth = spline_smooth(rx_w, ry_w, 0.5)

    path_xy = np.column_stack((rx_w_smooth, ry_w_smooth)) 



    distance = get_distance(rx_w, ry_w)
    print('distance = ',distance)
    # show_solution(grid, rx_w_smooth, ry_w_smooth)




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
    ref = PolylineReference(path, ds=0.10, v_ref=0.25)
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0) # Draw path
    


    # MPC setup
    Ts_mpc = 0.10
    N = 10 #Horizon
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    Q_matrix = np.diag([25.0, 25.0, 25.0])  # Position and heading tracking
    R_matrix = np.diag([0.1, 1.0])          # Control effort (v, w)
    P_matrix = np.diag([60.0, 60.0, 15.0])  # Terminal cost
    
    mpc = LinearMPCOSQP(
        Ts= Ts_mpc, 
        N= N,
        Q= Q_matrix,  
        R= R_matrix,  
        P= P_matrix,  
        vmin= 0.0,
        vmax= 1, 
        wmax= 4.0 )

    u_last = np.array([0.0, 0.0])
    
    history = []
    for t in range(n_steps): #basically for all t in simulation
        x = extract_base_state() #Extract pose

        # Update MPC at specified rate
        if t % steps_per_mpc == 0:
            x_ref, u_ref = ref.horizon(x[0], x[1], x[2], N)
            u_last, res = mpc.solve(x, x_ref, u_ref)
            
            # Print debug info every second
            # if t % (10 * steps_per_mpc) == 0:
            #     dx = x_ref[0, 0] - x[0]
            #     dy = x_ref[0, 1] - x[1]
            #     pos_error = np.sqrt(dx**2 + dy**2)
            #     theta_error = angle_difference(x_ref[0, 2], x[2])
                
            #     print(f"t={t*env.dt:4.1f}s: "
            #           f"pos=({x[0]:+5.2f}, {x[1]:+5.2f}), "
            #           f"θ={np.degrees(x[2]):+4.0f}°, "
            #           f"err={pos_error:.3f}m, "
            #           f"θ_err={np.degrees(theta_error):4.1f}°, "
            #           f"cmd=(v={u_last[0]:.2f}, w={u_last[1]:+.2f})")

        # Apply control
        action = build_action(env.n(), v=u_last[0], w=u_last[1])
        ob, reward, terminated, truncated, info = env.step(action)
        
        history.append((x.copy(), u_last.copy(), ob))  #useful information: True state, controll inputs, observations from simulation.
        
        if terminated or truncated:
            print(f"\n Terminated or Truncated at step {t}, see if collided or reached goal")
            break

    # Final results
    x_final = extract_base_state()
    goal_pos = path[-1]
    dist_to_goal = np.linalg.norm([x_final[0] - goal_pos[0], x_final[1] - goal_pos[1]])
    
    clear_debug_items(path_ids)
    env.close()
    return history

    # print(f"\n{'='*60}")
    # print(f"Simulation completed:")
    # print(f"  Final position: ({x_final[0]:.3f}, {x_final[1]:.3f})")
    # print(f"  Goal position:  ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
    # print(f"  Distance to goal: {dist_to_goal:.3f}m")
    # print(f"  Steps: {len(history)}")
    # print(f"{'='*60}\n")

    # Verify alignment
    # path_dir = np.arctan2(path[1,1] - path[0,1], path[1,0] - path[0,0])
    # alignment_err = angle_difference(path_dir, x0[2])
    
    # print(f"\nPath configuration:")
    # print(f"  Type: {path_type}, Length: {path_length}m")
    # print(f"  Start: ({path[0,0]:.3f}, {path[0,1]:.3f})")
    # print(f"  End:   ({path[-1,0]:.3f}, {path[-1,1]:.3f})")
    # print(f"  Path direction: {np.degrees(path_dir):.1f}°")
    # print(f"  Robot heading:  {np.degrees(x0[2]):.1f}°")
    # print(f"  Alignment: {np.degrees(alignment_err):.1f}° {'✓' if alignment_err < 0.1 else '✗'}")

    # print(f"\nMPC configuration:")
    # print(f"  Sample time: {Ts_mpc}s")
    # print(f"  Horizon: {N} steps")
    # print(f"  Updates every {steps_per_mpc} env steps")
    # print(f"{'='*60}\n")

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