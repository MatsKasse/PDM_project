from shapely.geometry import Polygon
import numpy as np
import pybullet as p

def _get_content_dict(obst):
    if hasattr(obst, "content_dict"):
        return obst.content_dict
    if hasattr(obst, "_content_dict"):
        return obst._content_dict
    raise AttributeError(f"Obstacle {type(obst)} has no content dict attribute")

def _box_to_polygon(cx, cy, w, l):
    hw, hl = w / 2.0, l / 2.0
    return Polygon([
        (cx - hw, cy - hl),
        (cx + hw, cy - hl),
        (cx + hw, cy + hl),
        (cx - hw, cy + hl),
    ])

def static_obstacles_to_circles(walls, boxes, cylinders,
                                robot_radius,
                                margin=0.2,
                                sample_radius=0.25,
                                spacing=None):
    if spacing is None:
        spacing = 1.5 * sample_radius

    circles = []

    # Cylinders -> 1 circle (inflated)
    for c in cylinders:
        d = _get_content_dict(c)
        x, y, _ = d["geometry"]["position"]
        r = float(d["geometry"]["radius"])
        circles.append((float(x), float(y), r + robot_radius + margin))

    # Boxes + walls -> boundary sampled circles (inflated polygon)
    for b in list(walls) + list(boxes):
        d = _get_content_dict(b)
        g = d["geometry"]
        cx, cy, _ = g["position"]
        l = float(g["width"])
        w = float(g["length"])
    

        poly = _box_to_polygon(float(cx), float(cy), w, l)
        inflated = poly.buffer(robot_radius + margin)

        boundary = inflated.exterior
        L = boundary.length
        s = 0.0
        while s < L:
            p = boundary.interpolate(s)
            circles.append((float(p.x), float(p.y), sample_radius))
            s += spacing

    return circles


def draw_circle_pybullet(x, y, r, z=0.05, color=[1, 0, 0], life_time=0, N=4):
    ids = []
    for i in range(N):
        th1 = 2*np.pi * i / N
        th2 = 2*np.pi * (i+1) / N
        p1 = [x + r*np.cos(th1), y + r*np.sin(th1), z]
        p2 = [x + r*np.cos(th2), y + r*np.sin(th2), z]
        ids.append(p.addUserDebugLine(p1, p2, color, lineWidth=1.5, lifeTime=life_time))
    return ids

def filter_circles_near_robot(static_circles, x, y, r_query=3.0):
    out = []
    rq2 = r_query*r_query
    for ox, oy, r in static_circles:
        if (ox-x)**2 + (oy-y)**2 <= rq2:
            out.append((ox, oy, r))
    return out

def filter_circles_near_robot_capped(static_circles, x, y, r_query=3.0, M_MAX=80):
    """
    1) filter binnen r_query
    2) sorteer op afstand tot robot
    3) neem maximaal M_MAX
    4) pad met dummy circles tot exact M_MAX (constante lengte!)
    """
    rq2 = r_query * r_query
    cand = []

    for ox, oy, r in static_circles:
        dx = ox - x
        dy = oy - y
        d2 = dx*dx + dy*dy
        if d2 <= rq2:
            cand.append((d2, ox, oy, r))

    # dichtstbij eerst
    cand.sort(key=lambda t: t[0])

    # cap
    out = [(ox, oy, r) for (_, ox, oy, r) in cand[:M_MAX]]

    # pad met “no-effect” obstacles ver weg
    # (ze komen nooit in de buurt → constraint wordt niet bindend)
    while len(out) < M_MAX:
        out.append((1e6, 1e6, 0.1))

    return out



