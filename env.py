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
rudder_angles = [0, 30, 60]


@ flow.math.jit_compile
def step(v, p, obj, rudder: flow.Obstacle, angle):
    obj = flow.advect.advect(obj, v, DT)
    # instantiate a new obstacle
    x, y = obj.geometry.center
    rudder_new = rudder.rotated(angle).at(
        flow.vec(x=x-(RUDDER_WIDTH*2)*flow.math.sin(angle),
                 y=y-(RUDDER_LENGTH_HALF)*flow.math.cos(angle)))

    v = flow.advect.semi_lagrangian(v, v, DT)
    v = v * (1 - BOUNDARY_MASK) + BOUNDARY_MASK * (SPEED, 0)
    v, p = flow.fluid.make_incompressible(
        v, [CYLINDER, rudder_new], flow.Solve('auto', 1e-5, x0=p))
    return v, p, obj, rudder_new


v_traj = [velocity]
p_traj = [point_cloud]
r_traj = [rudder]

for idx in tqdm(range(STEPS)):
    angle = 0
    if idx > 25:
        angle = rudder_angles[1]
    if idx > 50:
        angle = rudder_angles[2]
    if idx > 85:
        angle = rudder_angles[0]
    velocity, pressure, point_cloud, rudder_new = step(
        velocity, pressure, point_cloud, rudder, angle)
    v_traj.append(velocity)
    p_traj.append(point_cloud)
    r_traj.append(rudder_new)

traj = flow.stack(v_traj, flow.batch("time"))
p_traj = flow.stack(p_traj, flow.batch("time"))
r_traj = flow.stack(r_traj, flow.batch("time"))

anim = flow.plot([traj, CYLINDER_GEOM, r_traj.geometry, p_traj],
                 animate="time", size=(20, 10), overlay="list", frame_time=50,
                 color=["#43a5b3", "#f07857", "#ff87ce", "#bf2C34"])

# display the simulated results
plt.show()
anim.save("outputs/rudder_control.gif", writer="pillow")
