import warnings
import gymnasium as gym
import numpy as np
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.scene_examples.my_obstacles import *
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from PDM_project.mpc.reference_generator import PolylineReference, make_test_path



def run_albert(n_steps=1000, render=False, goal=True, obstacles=True):
    robots = [
        GenericDiffDriveRobot(
            urdf="albert.urdf",
            mode="vel",
            actuated_wheels=["wheel_right_joint", "wheel_left_joint"],
            castor_wheels=["rotacastor_right_joint", "rotacastor_left_joint"],
            wheel_radius = 0.08,
            wheel_distance = 0.494,
            spawn_rotation = 0,
            facing_direction = '-y',
        ),
    ]
    env: UrdfEnv = UrdfEnv(
        dt=0.01, robots=robots, render=render
    )
    action = np.zeros(env.n())
    action[0] = 0 ## forward speed
    action[1] = 0 ## rotational speed
    action[2] = 0 # joint 1
    action[3] = 0 # joint 2
    action[4] = 0 # joint 3
    action[5] = 0 # joint 4
    action[6] = 0 # joint 5
    action[7] = 0 # joint 6
    action[8] = 0 # joint 7
    action[9] = 0 # gripper open and close

    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )
    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    for cylinder in cylinder_obstacles:
        env.add_obstacle(cylinder)
    for box in box_obstacles:
        env.add_obstacle(box)

# --- Reference ---
    path = make_test_path("L")
    ref = PolylineReference(path, ds=0.10, v_ref=0.25)

        
    history = []
    for _ in range(n_steps):
        ob, *_ = env.step(action)
        history.append(ob)
    env.close()
    return history


if __name__ == "__main__":
    show_warnings = False
    warning_flag = "default" if show_warnings else "ignore"
    with warnings.catch_warnings():
        warnings.filterwarnings(warning_flag)
        run_albert(render=True)
