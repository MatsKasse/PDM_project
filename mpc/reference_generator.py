import numpy as np
import pybullet as p

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

class PolylineReference:
    """
    Houdt een polyline (waypoints) bij en levert een horizon [x_ref,y_ref,theta_ref]^T voor MPC.
    """
    def __init__(self, waypoints_xy: np.ndarray, ds=0.10, v_ref=0.25):
        assert waypoints_xy.shape[1] == 2
        self.ds = float(ds)
        self.v_ref = float(v_ref)
        self.path = self._resample_polyline(waypoints_xy, self.ds)

    @staticmethod
    def _resample_polyline(wp, ds):
        # eenvoudige resampling op vaste afstand
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
            out.append(wp[j] + t * (wp[j+1]-wp[j]))
        return np.array(out, dtype=float)

    def closest_index(self, x, y):
        d = self.path - np.array([x, y])
        return int(np.argmin(np.sum(d*d, axis=1)))

    def horizon(self, x, y, theta, N):
        """
        Return:
          xr: (N+1, 3)  -> [px, py, theta]
          ur: (N, 2)    -> [v_ref, w_ref] (hier w_ref=0; MPC mag sturen)
        """
        idx0 = self.closest_index(x, y)
        idxs = np.clip(idx0 + np.arange(N+1), 0, len(self.path)-1)

        pts = self.path[idxs]
        # heading via forward difference
        dpts = np.diff(pts, axis=0, prepend=pts[[0]])
        thetas = np.arctan2(dpts[:,1], dpts[:,0])
        thetas[0] = theta  # start consistent met huidige heading
        thetas = np.array([wrap_angle(t) for t in thetas])

        xr = np.column_stack([pts[:,0], pts[:,1], thetas])
        ur = np.column_stack([np.full(N, self.v_ref), np.zeros(N)])
        return xr, ur


def make_test_path(kind="L"):
    if kind == "straight":
        return np.array([[0.0, 1.0], [8.0, 1.0]])
    if kind == "L":
        return np.array([[0.0, 0.0], [0.0, -3.0], [-3.0, -3.0]])
    if kind == "S":
        xs = np.linspace(0, 8, 60)
        ys = 1.0 + 0.6*np.sin(0.7*xs)
        return np.column_stack([xs, ys])
    raise ValueError("Unknown path kind")




def draw_polyline(points_xy, z=0.02, line_width=2.0, life_time=0):
    """
    points_xy: array-like shape (M,2)
    life_time=0 => blijft staan
    returns: list met debug line ids
    """
    ids = []
    for i in range(len(points_xy) - 1):
        a = [float(points_xy[i][0]),   float(points_xy[i][1]),   z]
        b = [float(points_xy[i+1][0]), float(points_xy[i+1][1]), z]
        # kleur laat ik default; PyBullet pakt meestal wit/rood afhankelijk van client.
        ids.append(p.addUserDebugLine(a, b, lineWidth=line_width, lifeTime=life_time))
    return ids

def clear_debug_items(ids):
    for i in ids:
        try:
            p.removeUserDebugItem(i)
        except Exception:
            pass
