import math
import airsim
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from jsbsim_aircraft import Aircraft, x8
from jsbsim_simulator import Simulation
from autopilot import X8Autopilot
from image_processing import AirSimImages


class SingleObservation(gym.Env):
    """
    Description:
        An aircraft is placed at a predefined distance from an objective location at an airspeed defined by the
        aircraft's initial velocity.

    Source:
        This environment is based on the work of Alexander Quessy & Thomas Richardson

    Observation:
        Type: Box(1)
        Num     Observation     Min         Max
        0       RGB image       [0, 0, 0]   [255, 255, 255]

    Actions:
        Type: Box(2)
        [All dims in degrees]
        Num     Action      Min     Max
        0       Heading     -45     +45
        1       Pitch       -30     0

    Reward:
        Reward is 1 if step collides in a desired location
        Reward is 0 if step collides in undesirable location

    Starting State:
        Defined in basic_ic.xml

    Episode Termination:
        First time collision_info.has_collided == True
    """

    def __init__(self):

        self.max_sim_time: float = 100.0
        self.display_graphics: bool = True
        self.airspeed: float = 50.0
        self.airsim_frequency_hz: float = 48.0
        self.jsbsim_frequency_hz: float = 240.0
        self.aircraft: Aircraft = x8
        self.init_conditions = None
        self.debug_level: int = 0
        self.sim: Simulation = Simulation(self.jsbsim_frequency_hz,
                                          self.aircraft,
                                          self.init_conditions,
                                          self.debug_level)
        self.ap = X8Autopilot(self.sim)
        self.over: bool = False
        # angular limits
        self.max_hdg: float = 45.0
        self.min_hdg: float = -45.0
        self.max_pitch: float = 0.0
        self.min_pitch: float = -30.0
        max_angle = np.array([self.max_hdg, self.max_pitch], dtype=np.float32)
        min_angle = np.array([self.min_hdg, self.min_pitch], dtype=np.float32)
        self.action_space = spaces.Box(min_angle, max_angle, dtype=np.float32)
        self.images = AirSimImages(self.sim)
        dummy_obs = self.images.get_np_image(image_type=airsim.ImageType.Scene)
        self.observation_space = spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

        #  variables to keep track of step state
        self.graphic_update = 0
        self.max_updates = int(self.max_sim_time * self.jsbsim_frequency_hz)  # how many steps to update
        self.relative_update = self.airsim_frequency_hz / self.jsbsim_frequency_hz  # rate of airsim:JSBSim
        self.id = 0
        self.desired_heading = 0
        self.desired_pitch = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #  only update the commands if just starting the simulation
        self.desired_heading, self.desired_pitch = action
        obs = self.images.get_np_image(airsim.ImageType.Scene)
        # print(self.desired_pitch)
        graphic_i = self.relative_update * self.id
        graphic_update_old = self.graphic_update
        self.graphic_update = graphic_i // 1.0
        collision = self.sim.get_collision_info()
        done = False
        rewards = 0
        try:
            if collision.has_collided:
                done = True
                if collision.object_name == "airport":
                    rewards = 1
        except TypeError:
            print("Collision object does not exist yet")
        # update airsim if required
        if self.display_graphics and self.graphic_update > graphic_update_old:
            self.sim.update_airsim()
        # autopilot update
        self.ap.airspeed_hold_w_throttle(self.airspeed)
        self.ap.heading_hold(self.desired_heading)
        self.ap.pitch_hold(self.desired_pitch)
        self.sim.run()  # update jsbsim
        self.id = self.id + 1
        info = {}
        return obs, rewards, done, info

    def reset(self):
        """
        Reset the simulation to the initial state

        :return: state
        """
        self.graphic_update = 0
        self.max_updates = int(self.max_sim_time * self.jsbsim_frequency_hz)  # how many steps to update
        self.relative_update = self.airsim_frequency_hz / self.jsbsim_frequency_hz  # rate of airsim:JSBSim
        self.id = 0
        self.desired_heading = 0
        self.desired_pitch = 0
        self.sim.reinitialise()
        obs = self.images.get_np_image(image_type=airsim.ImageType.Scene)

        return obs

    def render(self, mode='human'):
        """
        Renders the graphics, this is done with UE4 and is integral to the programme
        :return:
        """

        return None
