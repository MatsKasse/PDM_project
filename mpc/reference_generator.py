import numpy as np
import pybullet as p


def wrap_angle(a):
    """ Makes sure the angle is always beteen [-pi, pi]"""
    return (a + np.pi) % (2 * np.pi) - np.pi


class PolylineReference:
    """
    Manages a polyline path and provides reference trajectories for MPC.
    Resamples the path to evenly dinstance points.
    Creates reference trajectories of these points for MPC horizon.
    """
    
    def __init__(self, waypoints_xy: np.ndarray, ds=0.10, v_ref=0.25):
        """
        Args:
            waypoints_xy: Array of shape (N, 2) with coordinates in x and y.
            ds: Distance between resampled points (meters)
            v_ref: Reference velocity (m/s)
        """
        assert waypoints_xy.shape[1] == 2 ##validation if waypoints are correct form
        self.ds = float(ds)
        self.v_ref = float(v_ref)
        self.path = self._resample_polyline(waypoints_xy, self.ds)
        self._last_idx = 0

        

    @staticmethod
    def _resample_polyline(wp, ds):
        """Resample waypoints at fixed distance intervals."""
        seg = wp[1:] - wp[:-1]      #create vectors between waypoints
        seglen = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seglen)]) # accumulated length of path
        total = s[-1]   #total path length
        
        if total < 1e-9:            # if path length is 0, return a copy of the waypoint and don't sample
            return wp.copy()
        
        s_new = np.arange(0.0, total + ds, ds)      # the path defined in desired steps ds
        out = []
        j = 0   #step zero (current)
        
        for si in s_new:
            while j < len(seglen) and s[j+1] < si:
                j += 1
            if j >= len(seglen):
                out.append(wp[-1])
                continue
            t = (si - s[j]) / max(seglen[j], 1e-9) 
            out.append(wp[j] + t * (wp[j+1] - wp[j]))   # this is the list of waypoints for horizon
        
        return np.array(out, dtype=float)

    def closest_index(self, x, y):
        """Find the point on the path that is closest to the robot."""
        d = self.path - np.array([x, y])
        idx = int(np.argmin(np.sum(d*d, axis=1)))
        idx = max(idx, self._last_idx)   # never go backwards
        self._last_idx = idx
        return idx


    def horizon(self, x, y, theta, N, use_sincos=False, use_shortest_angle=True, threshold=0.05):
        """
        Generate reference trajectory for MPC horizon.
            x, y: Current position
            theta: Current heading angle
            N: Horizon length
            use_sincos: Return [px, py, sin(theta), cos(theta)] if True
            use_shortest_angle: Align reference headings with current theta using atan2
        Returns:
            xr: Robot state (N+1, 3) with [px, py, theta] at each step
                or (N+1, 4) with [px, py, sin(theta), cos(theta)] if use_sincos
            ur: Robot input (N, 2) with [v, w] at each step
        """
        idx0 = self.closest_index(x, y) # Find closest point and get N+1 points ahead
        idxs = np.clip(idx0 + np.arange(N+1), 0, len(self.path)-1)  #make indices idx to idx0+N

        pts = self.path[idxs]   #select the N+1 reference points from the path
        
        # calculate the distance from the robot to the endpoint
        distance = np.linalg.norm(self.path[-1] - np.array([x, y]))
        slow_distance = 2
        stop_distance = threshold
        v_ref = self.v_ref * min(1.0, distance / slow_distance)
        if distance < stop_distance:
            v_ref = 0.0
        

        # Calculate vectors for every point in direction
        d = np.zeros_like(pts)
        d[:-1] = pts[1:] - pts[:-1]
        d[-1] = pts[-1] - pts[-2]   #last vector is from last point to the previous point just to give it a direction.
        
        thetas = np.arctan2(d[:,1], d[:,0])                 #Change vectors in to heading angles.
        # make heading continuous across ±pi
        thetas = np.unwrap(thetas)

        # align the first reference heading with current robot heading
        thetas = thetas - thetas[0] + theta

        if use_shortest_angle:
            # align each reference heading using atan2 shortest-angle difference
            delta = np.arctan2(np.sin(thetas - theta), np.cos(thetas - theta))
            thetas = theta + delta
        else:
            # optional: wrap back to [-pi, pi] to keep outputs bounded
            thetas = np.array([wrap_angle(t) for t in thetas])

        if use_sincos:
            s = np.sin(thetas)
            c = np.cos(thetas)
            xr = np.column_stack([pts[:,0], pts[:,1], s, c])
        else:
            xr = np.column_stack([pts[:,0], pts[:,1], thetas])  # state Vector of X, Y and theta

        ur = np.column_stack([np.full(N, v_ref), np.zeros(N)])     # imput vector v and w
        
        return xr, ur




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



def draw_polyline(points_xy, z=0.02, line_width=2.0, life_time=0):
    ids = []
    for i in range(len(points_xy) - 1):
        a = [float(points_xy[i][0]), float(points_xy[i][1]), z]
        b = [float(points_xy[i+1][0]), float(points_xy[i+1][1]), z]
        ids.append(p.addUserDebugLine(a, b, lineWidth=line_width, lineColorRGB=[0,0,1],lifeTime=life_time))
    return ids


def clear_debug_items(ids):
    """Remove debug visualization items."""
    for i in ids:
        try:
            p.removeUserDebugItem(i)
        except Exception:
            pass
