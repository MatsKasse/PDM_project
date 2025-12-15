import warnings
import gymnasium as gym
import numpy as np
import Gridmap as gm
import pybullet as p
#import A_star
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from my_obstacles import *


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
    action[0] = 0.2
    action[1] = 0.0
    action[2] = 0.5
    action[5] = -0.1
    ob = env.reset(
        pos=np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.5, 0.0, 1.8, 0.5])
    )

    for i in range(p.getNumBodies()):
        body_id = p.getBodyUniqueId(i)
        body_info = p.getBodyInfo(body_id)

        body_name = body_info[1].decode("utf-8")
        print(body_id, body_name)


    robot_id = 1

    aabb_min, aabb_max = p.getAABB(robot_id)

    dimensions = [
        aabb_max[0] - aabb_min[0],  # x size
        aabb_max[1] - aabb_min[1],  # y size
        aabb_max[2] - aabb_min[2],  # z size
    ]

    print("Robot dimensions (x, y, z):", dimensions)

    print(f"Initial observation : {ob}")
    for wall in wall_obstacles:
        env.add_obstacle(wall)
    for cylinder in cylinder_obstacles:
        env.add_obstacle(cylinder)
    for box in box_obstacles:
        env.add_obstacle(box)
    
    world_min, world_max = gm.get_world_bounds()
    x_min, y_min, _ = world_min
    x_max, y_max, _ = world_max

    print('x_min = ', x_min)
    print('x_max = ', x_max)

    gm.generate_gridmap(x_min, x_max, y_min, y_max, resolution=0.10)

    def get_action(iter):
        if iter % 50 == 0:
            action[0] += 0.1

        if iter > 500:
            action[0] = 0
            action[1] = 0.3
        return action
        
    print(f"Initial observation : {ob}")
    history = []
    for _ in range(n_steps):
        action = get_action(_)
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
