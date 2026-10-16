import numpy as np
import math

def choose_action(scene, history, rng):
    objs = []
    for name, o in scene.get("objects", {}).items():
        try:
            x, y = float(o["x"]), float(o["y"])
        except Exception:
            continue
        s = o.get("size", {})
        if "radius" in s:
            rr = abs(float(s["radius"]))
        else:
            vals = [abs(float(s.get(k, 1.0))) for k in
                    ("length", "thickness", "bottom_width", "top_width", "height")]
            rr = 0.5 * math.sqrt(sum(v*v for v in vals[:2]))
            if "height" in s:
                rr = 0.5 * math.sqrt(sum(v*v for v in vals[-2:]))
        rr = max(.08, min(2.0, rr))
        col = str(o.get("color", "")).lower()
        typ = str(o.get("type", "")).lower()
        objs.append((x, y, rr, col, typ, bool(o.get("dynamic", False))))

    green = [z for z in objs if "green" in z[3]]
    blue = [z for z in objs if "blue" in z[3]]
    g = green[0] if green else None
    b = blue[0] if blue else None

    def feasible(x, y, r):
        if x-r < -5 or x+r > 5 or y-r < -5 or y+r > 5:
            return False
        for ox, oy, orad, _, _, _ in objs:
            if (x-ox)*(x-ox) + (y-oy)*(y-oy) < (r+orad+.015)**2:
                return False
        return True

    anchors = []
    important = []
    for z in objs:
        if z[3] in ("green", "blue") or any(k in z[4] for k in
                                             ("lever", "ramp", "basket", "wall", "platform")):
            important.append(z)
    important += [z for z in objs if z not in important]

    for ox, oy, rr, _, _, _ in important:
        for a in rng.uniform(0, 2*math.pi, 5):
            d = rr + rng.uniform(.12, 1.2)
            anchors.append((ox + d*math.cos(a), oy + d*math.sin(a)))
    if g and b:
        gx, gy, gr = g[:3]
        bx, by, br = b[:3]
        dx, dy = bx-gx, by-gy
        dl = math.hypot(dx, dy) + 1e-8
        nx, ny = -dy/dl, dx/dl
        for t in rng.uniform(.15, .85, 8):
            px, py = gx+t*dx, gy+t*dy
            for side in (-1, 1):
                q = rng.uniform(.15, 1.2)
                anchors.append((px+side*nx*q, py+side*ny*q))
        anchors += [(gx, gy), (bx, by),
                    ((gx+bx)/2+nx*rng.uniform(-1,1),
                     (gy+by)/2+ny*rng.uniform(-1,1))]

    old = []
    for h in history[-20:]:
        a = h.get("action")
        if a is None or len(a) < 3:
            continue
        try:
            q = np.array([float(a[0]), float(a[1]), float(a[2])])
            bad = bool(h.get("invalid", False))
            dur = float(h.get("max_contact_duration", 0.0) or 0.0)
            sc = float(h.get("score", 0.0) or 0.0)
            suc = 1.0 if h.get("success", False) else 0.0
            val = 8*suc + dur/3.0 + .35*sc - 2.5*bad
            old.append((q, val, bad))
        except Exception:
            pass

    cand = []
    n = 150
    for i in range(n):
        r = float(np.clip(np.exp(rng.normal(math.log(.32), .48)), .11, 1.35))
        mode = rng.random()
        if old and mode < min(.55, .18 + .045*len(old)):
            q0, _, _ = old[int(rng.integers(len(old)))]
            sig = .18 + .55*rng.random()
            x = q0[0] + rng.normal(0, sig)
            y = q0[1] + rng.normal(0, sig)
            r = float(np.clip(q0[2] + rng.normal(0, .12), .11, 1.35))
        elif anchors and mode < .78:
            ax, ay = anchors[int(rng.integers(len(anchors)))]
            x = ax + rng.normal(0, .35 + .55*rng.random())
            y = ay + rng.normal(0, .35 + .55*rng.random())
        else:
            x, y = rng.uniform(-4.8, 4.8, 2)
        if feasible(x, y, r):
            cand.append(np.array([x, y, r]))

    if not cand:
        for _ in range(300):
            q = np.array([rng.uniform(-4.7, 4.7), rng.uniform(-4.7, 4.7),
                          rng.uniform(.11, .65)])
            if feasible(*q):
                cand.append(q)
                break
    if not cand:
        return [0.0, 0.0, 0.1]

    def prior(q):
        val = 0.0
        if g:
            d = math.hypot(q[0]-g[0], q[1]-g[1])
            val += 1.3*math.exp(-d/(q[2]+g[2]+.35))
        if b:
            d = math.hypot(q[0]-b[0], q[1]-b[1])
            val += 1.0*math.exp(-d/(q[2]+b[2]+.35))
        if g and b:
            mx, my = (g[0]+b[0])/2, (g[1]+b[1])/2
            d = math.hypot(q[0]-mx, q[1]-my)
            val += .8*math.exp(-d/1.7)
        return val

    scores = []
    for q in cand:
        s = prior(q) + rng.normal(0, .045)
        if old:
            ds = []
            for p, v, bad in old:
                z = (q-p) / np.array([2.5, 2.5, .55])
                w = math.exp(-float(np.dot(z, z))/2)
                ds.append((w, v, bad))
            den = sum(w for w, _, _ in ds)
            if den:
                pred = sum(w*v for w, v, _ in ds)/den
                badmass = sum(w for w, _, bad in ds if bad)/den
                s += (.16 + .035*len(old))*pred - .35*badmass
        scores.append(s)
    return cand[int(np.argmax(scores))].tolist()
