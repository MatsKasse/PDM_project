from shapely.geometry import Polygon

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

def filter_circles_near_robot(static_circles, x, y, r_query=4.0):
    out = []
    rq2 = r_query*r_query
    for ox, oy, r in static_circles:
        if (ox-x)**2 + (oy-y)**2 <= rq2:
            out.append((ox, oy, r))
    return out



