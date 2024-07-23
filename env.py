"""Karman Vortex Street
Air flow around a static cylinder.
Vortices start appearing after a couple of hundred steps.
"""

from phi.jax import flow
from tqdm import tqdm
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

DT = 0.5
STEPS = 100
'''
rudder logic:
the boat can be a point cloud object.
the rudder will simply act as an obstacle which spawns below the boat
'''
ORIGIN_X = 32
ORIGIN_Y = 10
RUDDER_WIDTH = 0.25
RUDDER_LENGTH = 2.5
RUDDER_LENGTH_HALF = RUDDER_LENGTH / 2

point_cloud = flow.PointCloud(flow.vec(x=ORIGIN_X, y=ORIGIN_Y))
rudder = flow.Obstacle(flow.geom.Box(
    x=(ORIGIN_X-RUDDER_WIDTH, ORIGIN_X+RUDDER_WIDTH),
    y=(ORIGIN_Y-RUDDER_LENGTH, ORIGIN_Y)))


@ flow.math.jit_compile
def step(v, p, obj, rudder: flow.Obstacle):
    obj = flow.advect.advect(obj, v, DT)
    # instantiate a new obstacle
    x, y = obj.geometry.center
    rudder = rudder.at(flow.vec(x=x, y=y-RUDDER_LENGTH_HALF))

    v = flow.advect.semi_lagrangian(v, v, DT)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    v, p = flow.fluid.make_incompressible(
        v, [CYLINDER, rudder], flow.Solve('auto', 1e-5, x0=p))
    return v, p, obj, rudder


v_traj = [velocity]
p_traj = [point_cloud]
r_traj = [rudder]

for _ in tqdm(range(STEPS)):
    velocity, pressure, point_cloud, rudder = step(
        velocity, pressure, point_cloud, rudder)
    v_traj.append(velocity)
    p_traj.append(point_cloud)
    r_traj.append(rudder)

traj = flow.stack(v_traj, flow.batch("time"))
p_traj = flow.stack(p_traj, flow.batch("time"))
r_traj = flow.stack(r_traj, flow.batch("time"))

anim = flow.plot([traj, CYLINDER_GEOM, r_traj.geometry, p_traj],
                 animate="time", size=(20, 10), overlay="list", frame_time=50,
                 color=["#43a5b3", "#f07857", "#ff87ce", "#bf2C34"])

# display the simulated results
plt.show()
anim.save("outputs/rudder_no_control_1.gif", writer="pillow")
