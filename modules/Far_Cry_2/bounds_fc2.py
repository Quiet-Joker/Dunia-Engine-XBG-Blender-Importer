from typing import List, Tuple
try:
    from ..Core.debug import VerboseLogger as vlog
except:
    class vlog:
        @staticmethod
        def log(m): pass
class BoundingBox:
    __slots__ = ('min', 'max')
    def __init__(self):
        # Two separate lists — aliasing min and max to one list silently
        # propagates any in-place mutation of one to the other.
        self.min = [0, 0, 0]
        self.max = [0, 0, 0]
class BoundingSphere:
    __slots__ = ('center', 'radius')
    def __init__(self):self.center = [0, 0, 0];self.radius = 0.0
def parse_xobb(g, chunk_size):
    try:
        g.i(2);min_max = g.f(6)
        if all(abs(v) < 100000 for v in min_max):
            bbox = BoundingBox()
            bbox.min = list(min_max[:3]);bbox.max = list(min_max[3:6])
            vlog.log(f"\nXOBB: min=({bbox.min[0]:.3f},{bbox.min[1]:.3f},{bbox.min[2]:.3f}) max=({bbox.max[0]:.3f},{bbox.max[1]:.3f},{bbox.max[2]:.3f})")
            return bbox
        return None
    except:return None
def parse_hpsb(g, chunk_size):
    try:
        g.i(2);sphere_data = g.f(4)
        if all(abs(v) < 100000 for v in sphere_data):
            sphere = BoundingSphere()
            sphere.center = list(sphere_data[:3]);sphere.radius = sphere_data[3]
            vlog.log(f"\nHPSB: center=({sphere.center[0]:.3f},{sphere.center[1]:.3f},{sphere.center[2]:.3f}) radius={sphere.radius:.3f}")
            return sphere
        return None
    except:return None
def clamp_to_16bit(value):return max(-32768, min(32767, value))
def check_bounds_exceeded(vertices, pos_scale):
    max_value = 32767
    inv_scale = 1.0 / pos_scale
    max_x = max_y = max_z = 0
    for vx, vy, vz in vertices:
        max_x = max(max_x, abs(int(vx * inv_scale)))
        max_y = max(max_y, abs(int(vy * inv_scale)))
        max_z = max(max_z, abs(int(vz * inv_scale)))
    max_coord = max(max_x, max_y, max_z)
    if max_coord > max_value:
        scale_factor = max_value / max_coord
        axis = "X" if max_coord == max_x else ("Y" if max_coord == max_y else "Z")
        return True, scale_factor, f"{axis} axis: {max_coord} (limit: {max_value})"
    return False, 1.0, "All coordinates within bounds"
