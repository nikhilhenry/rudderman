import numpy as np
import gymnasium as gym
from gymnasium import spaces
from phi import flow
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches


class KarmanVortexStreetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.size = (128, 64)  # length and height of the sim env
        self.bonus = 30000

        # for flow blind boat only the relative position is provided as an observation
        self.observation_space = spaces.Box(0, self.size[0], shape=(2,))

        # continous actions space of degree of rudder rotation from 0 to 360
        # should these be in radians instead? should these normalised

        self.action_space = spaces.Box(0, 360)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        Defining the phiflow constants
        """
        self._SPEED = 2
        self._DT = 0.25
        self._OBSTACLE_DIAMETER = 10
        BOUNDARY_BOX = flow.Box(x=(-1 * flow.INF, 0.5), y=None)
        self._BOUNDS = flow.Box(x=self.size[0], y=self.size[1])
        self._CYLINDER_X = 15
        self._CYLINDER_Y = 32
        self._CYLINDER_GEOM = flow.geom.infinite_cylinder(
            x=self._CYLINDER_X, y=self._CYLINDER_Y, radius=self._OBSTACLE_DIAMETER / 2
        )
        self._CYLINDER = flow.Obstacle(self._CYLINDER_GEOM)
        self._BOUNDARY_MASK = flow.StaggeredGrid(
            BOUNDARY_BOX, flow.ZERO_GRADIENT, x=128, y=128, bounds=self._BOUNDS
        )
        # defining rudder geometry
        RUDDER_WIDTH = 0.5
        RUDDER_LENGTH = 2.5
        self._RUDDER_LENGTH_HALF = RUDDER_LENGTH / 2
        self._RUDDER_GEOM = flow.geom.Box(x=RUDDER_WIDTH, y=RUDDER_LENGTH)


        # setting up for matplotlib rendering
        self.fig = None
        self.ax = None

    def _get_obs(self):
        x, y = self._boat.geometry.center
        return [x, y]

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                flow.math.numpy(
                    flow.math.vec_length(
                        self._boat.geometry.center - self._target_position
                    )
                )
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._step_count = 0

        # @todo randomly sample the start and target
        #start_x = self._CYLINDER_X + 5 * self._OBSTACLE_DIAMETER
        #start_y = self._CYLINDER_Y - 2.05 * self._OBSTACLE_DIAMETER
        start_x = 64
        start_y =  10
        target_x = self._CYLINDER_X + 5 * self._OBSTACLE_DIAMETER
        target_y = self._CYLINDER_Y + 2.05 * self._OBSTACLE_DIAMETER
        _start_position = flow.vec(x=start_x, y=start_y)
        self._target_position = flow.vec(x=target_x, y=target_y)

        """
        Setting up the PhiFlow sim environment
        """
        self._velocity = flow.StaggeredGrid(
            (self._SPEED, 0), flow.ZERO_GRADIENT, x=128, y=128, bounds=self._BOUNDS
        )
        self._pressure = None
        self._boat = flow.PointCloud(_start_position)

        self._prev_distance = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self,angle=0):
        if self.render_mode == "rgb_array":
            return self._render_frame(angle)

    def _sim_step(self, v, p, obj, angle: float):
        """
        accepts angles in radians
        """
        obj = flow.advect.advect(obj, v, self._DT)
        x, y = obj.geometry.center
        positon = flow.vec(x=x, y=y - self._RUDDER_LENGTH_HALF)
        rudder = self._RUDDER_GEOM.at(positon)
        rudder = flow.geom.rotate(rudder, angle, flow.vec(x=x, y=y))
        rudder = flow.Obstacle(rudder)

        v = flow.advect.semi_lagrangian(v, v, self._DT)
        v = v * (1 - self._BOUNDARY_MASK) + self._BOUNDARY_MASK * (self._SPEED, 0)
        v, p = flow.fluid.make_incompressible(
            v, [self._CYLINDER, rudder], flow.Solve("auto", 1e-5, x0=p)
        )
        return v, p, obj, rudder

    def step(self, action):
        self._step_count += 1

        # convert the action angle to radians
        angle = flow.math.degrees_to_radians(action)
        self._velocity, self._pressure, self._boat, _ = self._sim_step(
            self._velocity, self._pressure, self._boat, angle
        )

        observation = self._get_obs()
        info = self._get_info()

        # check to see that the boat is still in the world
        [x,y] = self._boat.geometry.center.numpy()
        if (
            x > self.size[0]
            and x < 0
            and y > self.size[1]
            and y < 0
        ):
            terminated = True
            reward = -1 * self._step_count
            return observation, reward, terminated, False, info

        # check euclidean distance between boat's position and target
        distance_to_target = flow.math.vec_length(
            self._boat.geometry.center - self._target_position
        )
        terminated = (
            True if distance_to_target <= self._OBSTACLE_DIAMETER / 6 else False
        )
        bonus = self.bonus if terminated else 0
        reward = (
            -1 * self._step_count
            + 10 * (distance_to_target - self._prev_distance)
            + bonus
        )

        if self.render_mode == "human":
            print("need to render")
            self._render_frame(angle)

        return observation, reward, terminated, False, info

    def _render_frame(self, angle):
        d = flow.plot([self._velocity,self._CYLINDER_GEOM,self._boat],size=(10,5),overlay="list")
        plt.draw()
        plt.pause(1)
        plt.clf()

if __name__ == "__main__":
    env = KarmanVortexStreetEnv(render_mode="human")
    observation = env.reset()
    for i in range(5):
        action = env.action_space.sample()[0]
        print(f"Action: {action}")
        observation, reward, done, _,info = env.step(action)
    env.close()
