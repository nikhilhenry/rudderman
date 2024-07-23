"""Karman Vortex Street
Air flow around a static cylinder.
Vortices start appearing after a couple of hundred steps.
"""

from phi.jax import flow
from tqdm import trange
from matplotlib import pyplot as plt

SPEED = 2
velocity = flow.StaggeredGrid(
    (SPEED, 0), flow.ZERO_GRADIENT, x=128, y=128, bounds=flow.Box(x=128, y=64))
CYLINDER_GEOM = flow.geom.infinite_cylinder(
    x=15, y=32, radius=10, inf_dim=None)
CYLINDER = flow.Obstacle(CYLINDER_GEOM)
BOUNDARY_BOX = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
BOUNDARY_MASK = flow.StaggeredGrid(
    BOUNDARY_BOX, flow.ZERO_GRADIENT, velocity.bounds, velocity.resolution)
pressure = None

'''
rudder logic:
the boat can be a point cloud object.
the rudder will simply act as an obstacle which spawns below the boat
'''
point_cloud = flow.PointCloud(flow.vec(x=5, y=5))
# print(point_cloud.geometry.center)
RUDDER_WIDTH = 2
RUDDER_LENGTH = 5
rudder_angle = 60

rudder = flow.Obstacle(flow.geom.Cuboid(
    flow.vec(x=5, y=5), x=RUDDER_WIDTH, y=RUDDER_LENGTH))


@flow.math.jit_compile
def step(v, p, obj):
    obj = flow.advect.advect(obj, v, 1.)
    # instantiate a new obstacle
    # x, y = obj.geometry.center
    v = flow.advect.semi_lagrangian(v, v, 1.)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    v, p = flow.fluid.make_incompressible(
        v, [CYLINDER], flow.Solve('auto', 1e-5, x0=p))
    return v, p, obj


traj, _, obj_traj = flow.iterate(step, flow.batch(time=5), velocity,
                                 pressure, point_cloud, range=trange)
anim = flow.plot([traj, CYLINDER_GEOM, obj_traj, rudder.geometry], animate="time",
                 size=(20, 10), overlay="list", frame_time=50)

# display the simulated results
plt.show()
