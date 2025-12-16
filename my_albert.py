import warnings
import numpy as np
import pybullet as p

from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv

from mpc.reference_generator import PolylineReference, make_test_path, draw_polyline, clear_debug_items, wrap_angle
from mpc.albert_control import extract_base_state, build_action, set_robot_body_id
from mpc.mpc_osqp import LinearMPCOSQP


# def find_robot_body_id():
#     """Find Albert robot body ID in PyBullet."""
#     for i in range(p.getNumBodies()):
#         body_info = p.getBodyInfo(i)
#         body_name = body_info[1].decode('utf-8').lower()
#         if any(keyword in body_name for keyword in ['albert', 'robot', 'mobile']):
#             print(f"Found robot: body {i} = '{body_info[1].decode('utf-8')}'")
#             return i
    
#     print("Robot not found by name, searching by structure...")
#     for i in range(p.getNumBodies()):
#         num_joints = p.getNumJoints(i)
#         if num_joints >= 4:
#             return i
    
#     return 0 ##Body is 1


def create_aligned_path(x0, y0, theta0, path_type="straight", length=3.0):
    """
    Create a path aligned with the robot's initial heading.
    """
    if path_type == "straight":
        local_path = np.array([[0.0, 0.0], [length, 0.0]])
    elif path_type == "L":
        local_path = np.array([
            [0.0, 0.0],
            [length, 0.0],
            [length, length]
        ])
    elif path_type == "S":
        x = np.linspace(0, length, 60)
        y = 0.4 * np.sin(2 * np.pi * x / length)
        local_path = np.column_stack([x, y])
    else:
        raise ValueError(f"Unknown path type: {path_type}")
    
    # Rotate to match robot heading
    cos_th = np.cos(theta0)
    sin_th = np.sin(theta0)
    rotation_matrix = np.array([[cos_th, -sin_th],
                                [sin_th,  cos_th]])
    
    rotated_path = local_path @ rotation_matrix.T
    global_path = rotated_path + np.array([x0, y0])
    
    return global_path


def angle_difference(angle1, angle2):
    """Calculate the smallest absolute difference between two angles."""
    diff = wrap_angle(angle1 - angle2)
    return abs(diff)


def run_albert(n_steps=1000, render=False, path_type="straight", path_length=3.0):
    """
    Run Albert robot simulation with MPC path following.
    
    Args:
        n_steps: Number of simulation steps
        render: Whether to show visualization
        path_type: Type of path ("straight", "L", "S")
        path_length: Length of path (meters)
    """
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius=0.08,
            wheel_distance=0.494,
            spawn_rotation=0,
            facing_direction='-y',
        ),
    ]
    
    env: UrdfEnv = UrdfEnv(dt=0.01, robots=robots, render=render)

    # Add obstacles
    # for wall in wall_obstacles:
    #     env.add_obstacle(wall)
    # for cylinder in cylinder_obstacles:
    #     env.add_obstacle(cylinder)
    # for box in box_obstacles:
    #     env.add_obstacle(box)

    # Reset environment
    ob, info = env.reset(pos=np.array([0.0, 0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5]))
    
    # Find and set robot body ID
    # robot_id = find_robot_body_id()
    robot_id = 1
    set_robot_body_id(robot_id)
    
    # Get initial state (now correctly returns -90° for -y facing robot)
    x0 = extract_base_state()
    print(f"\n{'='*60}")
    print(f"Initial state: pos=({x0[0]:.3f}, {x0[1]:.3f}), theta={np.degrees(x0[2]):.1f}°")

    # Create path aligned with robot's actual heading
    path = create_aligned_path(x0[0], x0[1], x0[2], path_type, path_length)
    ref = PolylineReference(path, ds=0.10, v_ref=0.25)
    
    # # Verify alignment
    # path_dir = np.arctan2(path[1,1] - path[0,1], path[1,0] - path[0,0])
    # alignment_err = angle_difference(path_dir, x0[2])
    
    # print(f"\nPath configuration:")
    # print(f"  Type: {path_type}, Length: {path_length}m")
    # print(f"  Start: ({path[0,0]:.3f}, {path[0,1]:.3f})")
    # print(f"  End:   ({path[-1,0]:.3f}, {path[-1,1]:.3f})")
    # print(f"  Path direction: {np.degrees(path_dir):.1f}°")
    # print(f"  Robot heading:  {np.degrees(x0[2]):.1f}°")
    # print(f"  Alignment: {np.degrees(alignment_err):.1f}° {'✓' if alignment_err < 0.1 else '✗'}")
    
    # Draw path
    path_ids = draw_polyline(ref.path, z=0.1, line_width=6.0, life_time=0)

    # MPC setup
    Ts_mpc = 0.10
    N = 10
    steps_per_mpc = int(round(Ts_mpc / env.dt))
    
    print(f"\nMPC configuration:")
    print(f"  Sample time: {Ts_mpc}s")
    print(f"  Horizon: {N} steps")
    print(f"  Updates every {steps_per_mpc} env steps")
    print(f"{'='*60}\n")

    mpc = LinearMPCOSQP(
        Ts=Ts_mpc, 
        N=N,
        Q=np.diag([40.0, 40.0, 10.0]),  # Position and heading tracking
        R=np.diag([0.1, 1.0]),          # Control effort (v, w)
        P=np.diag([60.0, 60.0, 15.0]),  # Terminal cost
        vmin=0.0,
        vmax=0.6, 
        wmax=1.0
    )

    u_last = np.array([0.0, 0.0])
    history = []
    
    for t in range(n_steps):
        x = extract_base_state()

        # Update MPC at specified rate
        if t % steps_per_mpc == 0:
            x_ref, u_ref = ref.horizon(x[0], x[1], x[2], N)
            u_last, res = mpc.solve(x, x_ref, u_ref)
            
            # Print debug info every second
            if t % (10 * steps_per_mpc) == 0:
                dx = x_ref[0, 0] - x[0]
                dy = x_ref[0, 1] - x[1]
                pos_error = np.sqrt(dx**2 + dy**2)
                theta_error = angle_difference(x_ref[0, 2], x[2])
                
                print(f"t={t*env.dt:4.1f}s: "
                      f"pos=({x[0]:+5.2f}, {x[1]:+5.2f}), "
                      f"θ={np.degrees(x[2]):+4.0f}°, "
                      f"err={pos_error:.3f}m, "
                      f"θ_err={np.degrees(theta_error):4.1f}°, "
                      f"cmd=(v={u_last[0]:.2f}, w={u_last[1]:+.2f})")

        # Apply control
        action = build_action(env.n(), v=u_last[0], w=u_last[1])
        ob, reward, terminated, truncated, info = env.step(action)
        
        history.append((x.copy(), u_last.copy(), ob))
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {t}")
            break

    # Final results
    x_final = extract_base_state()
    goal_pos = path[-1]
    dist_to_goal = np.linalg.norm([x_final[0] - goal_pos[0], x_final[1] - goal_pos[1]])
    
    print(f"\n{'='*60}")
    print(f"Simulation completed:")
    print(f"  Final position: ({x_final[0]:.3f}, {x_final[1]:.3f})")
    print(f"  Goal position:  ({goal_pos[0]:.3f}, {goal_pos[1]:.3f})")
    print(f"  Distance to goal: {dist_to_goal:.3f}m")
    print(f"  Steps: {len(history)}")
    print(f"{'='*60}\n")

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
            path_type="L",  # Options: "straight", "L", "S"
            path_length=3.0
        )