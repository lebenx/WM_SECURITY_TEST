import numpy as np
import math

def choose_action(scene, history, rng):
    objs = scene.get("objects", {})
    def find(color=None, typ=None, name=None):
        if name in objs:
            return objs[name]
        for v in objs.values():
            if (color is None or v.get("color") == color) and \
               (typ is None or v.get("type") == typ):
                return v
        return None

    green = find("green")
    blue = find("blue")
    red = find("red")
    gp = objs.get("gray_platform")
    ramp = objs.get("ramp")
    top = objs.get("top_basket")
    parcel = ramp is not None or top is not None

    valid_hist = [h for h in history if not h.get("invalid", False)]
    successes = [h for h in valid_hist if h.get("success", False)]
    if successes:
        h = max(successes, key=lambda z: z.get("score", 0.0))
        a = h.get("action")
        if a and rng.random() < 0.72:
            return [float(a[0]), float(a[1]), float(a[2])]

    best = None
    if valid_hist:
        best = max(valid_hist, key=lambda z: z.get("score", -1.0))
    n = len(history)

    def val(o, key, default=0.0):
        return float(o.get(key, default)) if o else default

    def candidate_ok(x, y, r):
        if x-r < -5 or x+r > 5 or y-r < -5 or y+r > 5:
            return False
        for o in objs.values():
            ox, oy = val(o, "x"), val(o, "y")
            s = o.get("size", {})
            typ = o.get("type", "")
            if typ == "Ball":
                rr = val(s, "radius", 0.3)
                if math.hypot(x-ox, y-oy) < r + rr + 0.035:
                    return False
            elif typ == "Bar":
                ln = val(s, "length", 1.0) * 0.5
                th = val(s, "thickness", 0.2) * 0.5
                ang = val(o, "angle")
                ca, sa = math.cos(ang), math.sin(ang)
                dx, dy = x-ox, y-oy
                u = dx*ca + dy*sa
                v = -dx*sa + dy*ca
                u = max(-ln, min(ln, u))
                v = max(-th, min(th, v))
                if math.hypot(dx-(u*ca-v*sa), dy-(u*sa+v*ca)) < r+0.04:
                    return False
            elif typ == "Basket":
                w = max(val(s, "top_width", 1.0),
                        val(s, "bottom_width", 1.0)) * 0.5
                hh = val(s, "height", 1.0) * 0.5
                if abs(x-ox) < r+w and abs(y-oy) < r+hh:
                    return False
        return True

    if parcel:
        tx = val(top, "x", val(green, "x"))
        ty = val(top, "y", val(green, "y"))
        use_ramp = False
        if ramp is not None:
            if n >= 3:
                old = [h for h in valid_hist if h.get("first_action_contact")]
                top_scores = [h.get("score", 0.0) for h in old
                              if "basket" in str(h.get("first_action_contact"))]
                ramp_scores = [h.get("score", 0.0) for h in old
                               if "ramp" in str(h.get("first_action_contact"))]
                use_ramp = (not ramp_scores and n >= 4) or (
                    top_scores and max(top_scores) < 0.9)
            elif n == 2:
                use_ramp = rng.random() < 0.35

        if best is not None and best.get("score", 0.0) > 0.82 and n >= 2:
            a = best.get("action", [])
            if len(a) == 3:
                x = a[0] + rng.normal(0, 0.24)
                y = a[1] + rng.normal(0, 0.28)
                r = a[2] + rng.normal(0, 0.08)
            else:
                x, y, r = tx + 0.6, ty + 2.65, 1.4
        elif use_ramp:
            ang = val(ramp, "angle")
            ln = val(ramp.get("size", {}), "length", 3.0)
            x = val(ramp, "x") - math.cos(ang)*ln*0.5
            y = val(ramp, "y") + math.sin(ang)*ln*0.5 + 2.0
            x += rng.normal(0, 0.22)
            y += rng.normal(0, 0.25)
            r = 1.25 + rng.normal(0, 0.13)
        else:
            x = tx + 0.62 + rng.normal(0, 0.28)
            y = max(ty + 2.45, val(green, "y") + 2.35)
            y += rng.normal(0, 0.32)
            r = 1.38 + rng.normal(0, 0.11)
    else:
        px = val(gp, "x", -1.0)
        py = val(gp, "y", -1.0)
        if best is not None and best.get("score", 0.0) > 0.62 and n >= 2:
            a = best.get("action", [])
            if len(a) == 3:
                x = a[0] + rng.normal(0, 0.20)
                y = a[1] + rng.normal(0, 0.30)
                r = a[2] + rng.normal(0, 0.07)
            else:
                x, y, r = px + 1.1, max(py + 1.9, val(green, "y") + 2.6), 1.4
        else:
            x = px + 1.08 + rng.normal(0, 0.30)
            y = max(py + 1.9, val(green, "y") + 2.55)
            y += rng.normal(0, 0.38)
            r = 1.38 + rng.normal(0, 0.10)

    r = max(0.18, min(1.5, r))
    x = max(-5+r+0.02, min(5-r-0.02, x))
    y = max(-5+r+0.02, min(5-r-0.02, y))

    offsets = [(0,0), (0,0.35), (0,-0.35), (-0.35,0),
               (0.35,0), (0.2,0.25), (-0.2,0.25)]
    for dx, dy in offsets:
        xx = max(-5+r+0.02, min(5-r-0.02, x+dx))
        yy = max(-5+r+0.02, min(5-r-0.02, y+dy))
        if candidate_ok(xx, yy, r):
            return [float(xx), float(yy), float(r)]

    r = min(r, 0.8)
    for k in range(20):
        xx = max(-5+r+0.03, min(5-r-0.03, x + rng.normal(0, 0.5)))
        yy = max(-5+r+0.03, min(5-r-0.03, y + rng.normal(0, 0.5)))
        if candidate_ok(xx, yy, r):
            return [float(xx), float(yy), float(r)]
    return [float(x), float(y), float(r)]
