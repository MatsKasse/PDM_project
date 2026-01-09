import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import math

# Generate a 2D gridmap of the environment using ray tests in pybullet
def generate_gridmap(xmin, xmax, ymin, ymax, resolution=1, robot_radius=0.4, ray_height=5.0, nr_dyn_obst=0, batch_size=100):
    w = int((xmax - xmin) / resolution)
    h = int((ymax - ymin) / resolution)

    grid = np.zeros((w, h), dtype=np.uint8)

    ray_starts = []
    ray_ends = []
    coords = []

    #add rays for each grid cell
    for i in range(w):
        for j in range(h):
            x = xmin + i * resolution
            y = ymin + j * resolution
            ray_starts.append([x, y, ray_height])
            ray_ends.append([x, y, -1])    # ray goes straight down
            coords.append((i, j))

    results = Batched_Ray_Test(ray_starts, ray_ends)

    #For each ray test result, mark grid cell as occupied or free (0=free, 1=occupied)
    for k, result in enumerate(results):
        object_id = result[0]
        hit = object_id > 1 + nr_dyn_obst #object_id '0' is the groundplane, object_id '1' is robot, if larger it is an obstacle. Returns -1 if no hit.
        i, j = coords[k]
        grid[i, j] = 1 if hit else 0

    #Transpose grid to have correct orientation
    grid = grid.T

    #Inflate obstacles in gridmap according to robot radius
    inflated_grid = inflate_grid(grid, resolution, robot_radius)

    return grid, inflated_grid

# Perform batched ray tests in pybullet to improve performance
def Batched_Ray_Test(ray_starts, ray_ends, batch_size=1000):
    results = []
    for i in range(0, len(ray_starts), batch_size):
        rs = ray_starts[i: i+batch_size]
        re = ray_ends[i: i+batch_size]
        results.extend(p.rayTestBatch(rs, re))

    return results

#Get the world bounds of all objects in the pybullet simulation
def get_world_bounds():
    n_bodies = p.getNumBodies()
    ground_id = 0
    padding = 0.5
    global_min = [float('inf')] * 3
    global_max = [-float('inf')] * 3

    for body_id in range(n_bodies):
        if body_id == ground_id:
            continue
        
        #Bounding box information of the object
        aabb_min, aabb_max = p.getAABB(body_id)

        for i in range(3):
            global_min[i] = min(global_min[i], aabb_min[i])
            global_max[i] = max(global_max[i], aabb_max[i])

    for i in range(3):
        global_min[i] -= padding
        global_max[i] += padding

    return global_min, global_max

def inflate_grid(grid, resolution, robot_radius):
        inflated = grid.copy()
        xs, ys = np.where(grid == 1)
        robot_radius = int(math.ceil(robot_radius / resolution))
        
        #All pixels within robot_radius of an obstacle are also marked as obstacles
        for x, y in zip(xs, ys):
            y_min = max(0, y - robot_radius)
            y_max = min(grid.shape[0], y + robot_radius + 1)
            x_min = max(0, x - robot_radius)
            x_max = min(grid.shape[1], x + robot_radius + 1)
            inflated[x_min:x_max, y_min:y_max] = 1

        return inflated