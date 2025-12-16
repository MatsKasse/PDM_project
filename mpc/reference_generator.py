import numpy as np
import pybullet as p


def wrap_angle(a):
    """Wrap angle to [-pi, pi] range."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class PolylineReference:
    """
    Manages a polyline path and provides reference trajectories for MPC.
    """
    
    def __init__(self, waypoints_xy: np.ndarray, ds=0.10, v_ref=0.25):
        """
        Args:
            waypoints_xy: Array of shape (N, 2) with path waypoints
            ds: Distance between resampled points (meters)
            v_ref: Reference velocity (m/s)
        """
        assert waypoints_xy.shape[1] == 2
        self.ds = float(ds)
        self.v_ref = float(v_ref)
        self.path = self._resample_polyline(waypoints_xy, self.ds)

    @staticmethod
    def _resample_polyline(wp, ds):
        """Resample waypoints at fixed distance intervals."""
        seg = wp[1:] - wp[:-1]
        seglen = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(seglen)])
        total = s[-1]
        
        if total < 1e-9:
            return wp.copy()
        
        s_new = np.arange(0.0, total + ds, ds)
        out = []
        j = 0
        
        for si in s_new:
            while j < len(seglen) and s[j+1] < si:
                j += 1
            if j >= len(seglen):
                out.append(wp[-1])
                continue
            t = (si - s[j]) / max(seglen[j], 1e-9)
            out.append(wp[j] + t * (wp[j+1] - wp[j]))
        
        return np.array(out, dtype=float)

    def closest_index(self, x, y):
        """Find index of closest point on path to (x, y)."""
        d = self.path - np.array([x, y])
        return int(np.argmin(np.sum(d*d, axis=1)))

    def horizon(self, x, y, theta, N):
        """
        Generate reference trajectory for MPC horizon.
        
        Args:
            x, y: Current position
            theta: Current heading angle
            N: Horizon length
        
        Returns:
            xr: Array of shape (N+1, 3) with [px, py, theta] at each step
            ur: Array of shape (N, 2) with [v, w] at each step
        """
        # Find closest point and get N+1 points ahead
        idx0 = self.closest_index(x, y)
        idxs = np.clip(idx0 + np.arange(N+1), 0, len(self.path)-1)

        pts = self.path[idxs]
        
        # Calculate heading angles from path direction
        d = np.zeros_like(pts)
        d[:-1] = pts[1:] - pts[:-1]
        d[-1] = pts[-1] - pts[-2]
        
        thetas = np.arctan2(d[:,1], d[:,0])
        
        # CRITICAL FIX: Use current heading for first reference point
        # This prevents the robot from making an initial turn
        thetas[0] = theta
        
        # Wrap all angles to [-pi, pi]
        thetas = np.array([wrap_angle(t) for t in thetas])

        xr = np.column_stack([pts[:,0], pts[:,1], thetas])
        ur = np.column_stack([np.full(N, self.v_ref), np.zeros(N)])
        
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
    """
    Draw a polyline in PyBullet for visualization.
    
    Args:
        points_xy: Array of shape (M, 2)
        z: Height above ground
        line_width: Width of debug line
        life_time: 0 = permanent
    
    Returns:
        List of debug line IDs
    """
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