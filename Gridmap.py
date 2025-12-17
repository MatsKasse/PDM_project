import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import math

display = False

def generate_gridmap(xmin, xmax, ymin, ymax, resolution=1, ray_height=5.0, batch_size=100):
    w = int((xmax - xmin) / resolution)
    h = int((ymax - ymin) / resolution)

    grid = np.zeros((w, h), dtype=np.uint8)  # 0=free, 1=occupied

    ray_starts = []
    ray_ends = []
    coords = []

    for i in range(w):
        for j in range(h):
            x = xmin + i * resolution
            y = ymax - j * resolution
            ray_starts.append([x, y, ray_height])
            ray_ends.append([x, y, -1])    # ray goes straight down
            coords.append((i, j))

    results = Batched_Ray_Test(ray_starts, ray_ends)

    for k, result in enumerate(results):
        object_id = result[0]
        hit = object_id > 0 #object_id '0' is the groundplane, if larger it is an object. Returns -1 if no hit.
        i, j = coords[k]
        grid[i, j] = 1 if hit else 0

    if display:
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.T, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        plt.show()

    ox, oy = obs_coords(grid, resolution)

    grid = grid.T
    inflated_grid = inflate_grid(grid, resolution)

    return grid, inflated_grid

def Batched_Ray_Test(ray_starts, ray_ends, batch_size=1000):
    results = []
    for i in range(0, len(ray_starts), batch_size):
        rs = ray_starts[i: i+batch_size]
        re = ray_ends[i: i+batch_size]
        results.extend(p.rayTestBatch(rs, re))

    return results

def get_world_bounds():
    n_bodies = p.getNumBodies()
    ground_id = 0
    padding = 0.0
    global_min = [float('inf')] * 3
    global_max = [-float('inf')] * 3

    for body_id in range(n_bodies):
        if body_id == ground_id:
            continue

        aabb_min, aabb_max = p.getAABB(body_id)

        for i in range(3):
            global_min[i] = min(global_min[i], aabb_min[i])
            global_max[i] = max(global_max[i], aabb_max[i])

    for i in range(3):
        global_min[i] -= padding
        global_max[i] += padding

    return global_min, global_max

def obs_coords(grid, resolution):
    w, h = grid.shape[:2]
    ox = []
    oy = []

    for i in range(w):
        for j in range(h):
            if grid[i, j] == 0:
                ox.append(i*resolution)
                oy.append(j*resolution)

    return ox, oy

def inflate_grid(grid, resolution):
        inflated = grid.copy()
        ys, xs = np.where(grid == 1)
        robot_radius = int(math.ceil(0.3 / resolution))

        for y, x in zip(ys, xs):
            y_min = max(0, y - robot_radius)
            y_max = min(grid.shape[0], y + robot_radius + 1)
            x_min = max(0, x - robot_radius)
            x_max = min(grid.shape[1], x + robot_radius + 1)
            inflated[x_min:x_max, y_min:y_max] = 1

        return inflated