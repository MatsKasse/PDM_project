import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import math

# Generate a 2D gridmap of the environment using ray tests in pybullet
def generate_gridmap(xmin, xmax, ymin, ymax, resolution=1, robot_radius=0.4, ray_height=5.0, batch_size=100):
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
        hit = False
        if object_id > 1:
            # Only count static, non-sphere bodies as obstacles in the gridmap.
            # Dynamic obstacles in this project are spheres, so we ignore spheres here.
            mass = p.getDynamicsInfo(object_id, -1)[0]
            if mass == 0:
                shape_data = p.getCollisionShapeData(object_id, -1)
                is_sphere = any(sd[2] == p.GEOM_SPHERE for sd in shape_data)
                hit = not is_sphere
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

#Convert world coordinates to grid coordinates
def world_to_grid(x, y, x_min, y_min, resolution):
    x_g = int((x - x_min) / resolution)
    y_g = int((y - y_min) / resolution)
    return x_g, y_g

#Convert grid coordinates to world coordinates
def grid_to_world(x_g, y_g, x_min, y_min, resolution):
    x = (x_g) * resolution + x_min
    y = (y_g) * resolution + y_min
    return x, y
