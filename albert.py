import warnings
import gymnasium as gym
import numpy as np
import Gridmap as gm
import pybullet as p
import random
from urdfenvs.robots.generic_urdf.generic_diff_drive_robot import GenericDiffDriveRobot
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from my_obstacles import *
from a_star import *

x_min = 0
y_min = 0

sx = 16.0
sy = 2.5
gx = 1.0
gy = 12.5

visits = 3

resolution = 0.09

Multiple_Points = False

def world_to_grid(x, y):
    x_g = int((x - x_min) / resolution)
    y_g = int((y - y_min) / resolution)
    return x_g, y_g

def grid_to_world(x_g, y_g):
    x = (x_g) * resolution + x_min
    y = (y_g) * resolution + y_min
    return x, y

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
    
        ax.imshow(grid, cmap='Greys', vmin=0, vmax=1, interpolation="nearest", origin='lower', extent=[0, grid.shape[1]*resolution, 0, grid.shape[0]*resolution])

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
    for box in box_obstacles:
        env.add_obstacle(box)
    
    world_min, world_max = gm.get_world_bounds()
    x_min, y_min, _ = world_min
    x_max, y_max, _ = world_max

    grid, inflated_grid = gm.generate_gridmap(x_min, x_max, y_min, y_max, resolution=resolution)
    A_star = AStarPlanner(resolution, 0.3, inflated_grid, x_min, y_min, x_max, y_max)

    sx_g, sy_g = world_to_grid(sx, sy)
    
    def get_path(gx, gy):
        gx = gx
        gy = gy

        if Multiple_Points:
            gx, gy = np.zeros(visits), np.zeros(visits)
            gx_g, gy_g = np.zeros(visits), np.zeros(visits)
            rx_w_list = []
            ry_w_list = []

            pot_x, pot_y = np.where(inflated_grid == 0)
            pos = np.column_stack((pot_x, pot_y))
            indices = np.random.choice(len(pos), size=visits, replace=False)
            
            for i, idx in enumerate(indices):
                gx_g[i], gy_g[i] = pos[idx]
                gx[i], gy[i] = grid_to_world(gx_g[i], gy_g[i])
                if i == 0:
                    rx_g, ry_g = A_star.planning(sx_g, sy_g, gx_g[i], gy_g[i])
                else:
                    rx_g, ry_g = A_star.planning(gx_g[i-1], gy_g[i-1], gx_g[i], gy_g[i])
                
                rx_w, ry_w = zip(*[grid_to_world(x, y) for x, y in zip(rx_g, ry_g)])
                rx_w_list += rx_w
                ry_w_list += ry_w

        else:
            gx_g, gy_g = world_to_grid(gx, gy)
            rx_g, ry_g = A_star.planning(sx_g, sy_g, gx_g, gy_g)
            rx_w_list, ry_w_list = zip(*[grid_to_world(x, y) for x, y in zip(rx_g, ry_g)])

        return rx_w_list, ry_w_list

    rx_w, ry_w = get_path(gx, gy)
    distance = get_distance(rx_w, ry_w)
    print(distance)
    show_solution(grid, rx_w, ry_w)
    print('rx:', rx_w)
    print('ry:', ry_w) 

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

