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
CYLINDER_GEOM = flow.geom.infinite_cylinder(x=15, y=32, radius=5, inf_dim=None)
CYLINDER = flow.Obstacle(CYLINDER_GEOM)
BOUNDARY_BOX = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
BOUNDARY_MASK = flow.StaggeredGrid(
    BOUNDARY_BOX, velocity.extrapolation, velocity.bounds, velocity.resolution)
pressure = None


@flow.math.jit_compile
def step(v, p):
    v = flow.advect.semi_lagrangian(v, v, 1.)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    v, p = flow.fluid.make_incompressible(
        v, [CYLINDER], flow.Solve('auto', 1e-5, x0=p))
    return v, p


traj, _ = flow.iterate(step, flow.batch(time=100), velocity,
                       pressure, range=trange)
anim = flow.plot([traj, CYLINDER_GEOM], animate="time",
                 size=(20, 10), overlay="list", frame_time=50)
plt.show()
