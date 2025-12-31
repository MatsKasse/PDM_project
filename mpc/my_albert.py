import warnings
import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpc.reference_generator import PolylineReference, draw_polyline, clear_debug_items, wrap_angle, create_aligned_path
from mpc.albert_control import extract_base_state, build_action, set_robot_body_id, angle_difference
from mpc.mpc_osqp import LinearMPCOSQP



def run_albert(n_steps=1000, render=False, path_type="straight", path_length=3.0):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
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

    
    
    robot_id = 1
    set_robot_body_id(robot_id)
    

    # Get initial state (now correctly returns -90° for -y facing robot)
    x0 = extract_base_state()
    print(f"\n{'='*60}")
    print(f"Initial state: pos=({x0[0]:.3f}, {x0[1]:.3f}), theta={np.degrees(x0[2]):.1f}°")

    # Create path aligned with robot's actual heading
    path = create_aligned_path(x0[0], x0[1], x0[2], path_type, path_length)
    ref = PolylineReference(path, ds=0.10, v_ref=1)
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0) # Draw path
    


    # MPC setup
    Ts_mpc = 0.10
    N = 10 #Horizon
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    Q_matrix = np.diag([40.0, 40.0, 10.0, 10.0])  # Position and sin/cos tracking
    R_matrix = np.diag([0.1, 1.0])          # Control effort (v, w)
    P_matrix = np.diag([60.0, 60.0, 15.0, 15.0])  # Terminal cost
    
    mpc = LinearMPCOSQP(
        Ts= Ts_mpc, 
        N= N,
        Q= Q_matrix,  
        R= R_matrix,  
        P= P_matrix,  
        vmin= 0.0,
        vmax= 7, 
        wmax= 4.0 )

    u_last = np.array([0.0, 0.0])
    
    history = []
    def state_to_sincos(x_state):
        return np.array([x_state[0], x_state[1], np.sin(x_state[2]), np.cos(x_state[2])], dtype=float)

    for t in range(n_steps): #basically for all t in simulation
        x = extract_base_state() #Extract pose
        x_mpc = state_to_sincos(x)

        # Update MPC at specified rate
        if t % steps_per_mpc == 0:
            x_ref, u_ref = ref.horizon(x[0], x[1], x[2], N, use_sincos=True, use_shortest_angle=True)
            u_last, res = mpc.solve(x_mpc, x_ref, u_ref)
            
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
