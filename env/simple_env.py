import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_checker import check_env
from phi.jax import flow
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
import gc

plt.rcParams["figure.figsize"] = (10, 5)

"""
Defining the phiflow constants
"""
_SIZE = (128, 64)  # length and height of the sim env
_OBSTACLE_DIAMETER = 10
_BOUNDARY_BOX = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
_BOUNDS = flow.Box(x=_SIZE[0], y=_SIZE[1])
_CYLINDER_X = 15
_CYLINDER_Y = 32
_BOUNDARY_MASK = flow.StaggeredGrid(
    _BOUNDARY_BOX, flow.ZERO_GRADIENT, x=128, y=128, bounds=_BOUNDS
)


_DT = 0.5
# defining rudder geometry
_RUDDER_WIDTH = 0.5
_RUDDER_LENGTH = 2.5
_RUDDER_LENGTH_HALF = _RUDDER_LENGTH / 2
_RUDDER_GEOM = flow.geom.Box(x=_RUDDER_WIDTH, y=_RUDDER_LENGTH)
_SPEED = 2


def scale_to_angle(x):
    return 180 * (x + 1)


def angle_to_scale(x):
    return x / 180 - 1


@flow.math.jit_compile
def _sim_step(v, p, obj, angle: float):
    """
    accepts angles in radians
    """
    obj = flow.advect.advect(obj, v, _DT)
    x, y = obj.geometry.center
    positon = flow.vec(x=x, y=y - _RUDDER_LENGTH_HALF)
    rudder = _RUDDER_GEOM.at(positon)
    rudder = flow.geom.rotate(rudder, angle, flow.vec(x=x, y=y))
    rudder = flow.Obstacle(rudder)

    v = flow.advect.semi_lagrangian(v, v, _DT)
    v = v * (1 - _BOUNDARY_MASK) + _BOUNDARY_MASK * (_SPEED, 0)
    v, p = flow.fluid.make_incompressible(v, [rudder], flow.Solve("auto", 1e-5, x0=p))
    return v, p, obj, rudder


class SimpleFlowEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.bonus = 300000

        # for flow blind boat only the relative position is provided as an observation
        self.observation_space = spaces.Box(0, 1, shape=(2,))

        # continous actions space of degree of rudder rotation from 0 to 360
        # should these be in radians instead? should these normalised

        self.action_space = spaces.Box(-1, 1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # setting up for matplotlib rendering
        self.fig = None
        self.ax = None

    def _get_obs(self):
        x, y = self._boat.geometry.center
        rel_x, rel_y = self._target_x - x, self._target_y - y
        REL_X_MAX = 140
        REL_X_MIN = -REL_X_MAX
        REL_Y_MAX = 70
        REL_Y_MIN = -REL_Y_MAX
        # normalise these values
        rel_x = (rel_x - REL_X_MIN) / (REL_X_MAX - REL_X_MIN)
        rel_y = (rel_y - REL_Y_MIN) / (REL_Y_MAX - REL_Y_MIN)
        return np.asarray([rel_x, rel_y], dtype=np.float32)

    def _get_info(self):
        return {
            "distance": flow.math.numpy(
                flow.math.vec_length(self._boat.geometry.center - self._target_position)
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        # @todo randomly sample the start and target
        self.start_x = _CYLINDER_X + 5 * _OBSTACLE_DIAMETER
        self.start_y = _CYLINDER_Y - 2.05 * _OBSTACLE_DIAMETER
        target_x = _CYLINDER_X + 5 * _OBSTACLE_DIAMETER
        target_y = _CYLINDER_Y + 2.05 * _OBSTACLE_DIAMETER
        _start_position = flow.vec(x=self.start_x, y=self.start_y)
        self._target_position = flow.vec(x=target_x, y=target_y)
        self._target_x = target_x
        self._target_y = target_y

        """
        Setting up the PhiFlow sim environment
        """
        self._velocity = flow.StaggeredGrid(
            (_SPEED, 0), flow.ZERO_GRADIENT, x=128, y=128, bounds=_BOUNDS
        )
        self._pressure = None
        self._boat = flow.PointCloud(_start_position)

        self._prev_distance = 0

        # storing this angle for rendering
        self._angle = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self, mode="rgb_array"):
        if self.render_mode == "rgb_array" or mode == "rgb_array":
            return self._render_frame(self._angle)

    def step(self, action):
        self._step_count += 1
        # scaling the action to degrees
        action = scale_to_angle(action[0])
        # convert the action angle to radians
        angle = flow.math.degrees_to_radians(action)
        self._angle = angle
        self._velocity, self._pressure, self._boat, _ = _sim_step(
            self._velocity, self._pressure, self._boat, angle
        )

        observation = self._get_obs()
        info = self._get_info()

        # check to see that the boat is still in the world
        [x, y] = self._boat.geometry.center.numpy()
        if x > _SIZE[0] or x < 0 or y > _SIZE[1] or y < 0:
            terminated = False
            truncated = True
            reward = 0
            return observation, reward, terminated, truncated, info

        if self._step_count > 350:
            terminated = False
            truncated = True
            reward = 0
            print("exceeded number of iterations")
            return observation, reward, terminated, truncated, info

        # check euclidean distance between boat's position and target
        distance_to_target = flow.math.vec_length(
            self._target_position - self._boat.geometry.center
        ).numpy()
        terminated = True if distance_to_target <= _OBSTACLE_DIAMETER / 6 else False
        bonus = self.bonus if terminated else 0
        move_score = np.max([(self._prev_distance - distance_to_target), 1])
        reward = 10 * move_score + bonus
        self._prev_distance = distance_to_target

        if self.render_mode == "human":
            self._render_frame(angle)

        return observation, reward, terminated, False, info

    def _render_frame(self, angle):
        # creating the rudder to be drawn
        x, y = self._boat.geometry.center
        positon = flow.vec(x=x, y=y - _RUDDER_LENGTH_HALF)
        rudder = _RUDDER_GEOM.at(positon)
        rudder = flow.geom.rotate(rudder, angle, flow.vec(x=x, y=y))
        # drawing vector field and rudder
        d = flow.plot(self._velocity, rudder, size=(10, 5), overlay="args")
        # drawing the boat
        boat = mpatches.Circle((x, y), 0.5, fc="k")
        plt.gca().add_patch(boat)

        # drawing the start and end positions
        start_circle = mpatches.Circle(
            (self.start_x, self.start_y),
            1,
            ec="xkcd:violet",
            fc="xkcd:violet",
            alpha=0.5,
            ls="--",
            lw=1,
        )
        plt.gca().add_patch(start_circle)
        target_circle = mpatches.Circle(
            (self._target_x, self._target_y),
            _OBSTACLE_DIAMETER / 6,
            ec="xkcd:red",
            fc="xkcd:red",
            alpha=0.5,
            ls="--",
            lw=1,
        )
        plt.gca().add_patch(target_circle)
        # configurations
        plt.xlim(0, 128)
        plt.ylim(0, 64)
        if self.render_mode == "human":
            plt.draw()
            plt.pause(0.5)
            plt.close()  #:FIXME: force phiflow to use the same figure and instead clear the figure
            return
        if self.render_mode == "rgb_array":
            canvas = FigureCanvasAgg(plt.gcf())
            canvas.draw()
            buf = canvas.buffer_rgba()
            plt.close()
            gc.collect()
            return np.asarray(buf)


if __name__ == "__main__":
    env = SimpleFlowEnv(render_mode="rgb_array")
    check_env(env, warn=True)
    env = RecordVideo(
        env,
        video_folder="./logs/vortex_outputs/simple_flow_env",
        name_prefix="eval",
        episode_trigger=lambda x: True,
    )
    observation = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()[0]
        print(f"Action: {scale_to_angle(action)}")
        observation, reward, done, _, info = env.step([action])
    print(observation, reward, done, info)
    env.close()
