from abc import ABC
import numpy as np
import airsim
import gym
# from tasks import Shaping
from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from debug_utils import *
import jsbsim_properties as prp
from simple_pid import PID
from autopilot import X8Autopilot
from navigation import WindEstimation
from image_processing import AirSimImages, SemanticImageSegmentation
from typing import Type, Tuple, Dict
from tasks import *


class Environment(gym.Env):
    """
    A class to run airsim, JSBSim and join the other classes together

    ...

    Attributes:
    ----------
    sim_time : float
        how many seconds to run the simulation for
    display_graphics : bool
        decides whether to run the airsim graphic update in unreal, required for image_processing input
    airspeed : float
        fixed airspeed used to fly the aircraft if airspeed_hold_w_throttle a/p used
    agent_interaction_frequency_hz : float
        how often the agent selects a new action, should be equal to or the lowest frequency
    airsim_frequency_hz : float
        how often to update the airsim graphic simulation
    sim_frequency_hz : float
        how often to update the JSBSim input, should not be less than 120Hz to avoid unexpected behaviour
    aircraft : Aircraft
        the aircraft type used, x8 by default, changing this will likely require a change in the autopilot used
    init_conditions : Dict[prp.Property, float] = None
        the simulations initial conditions None by default as in basic_ic.xml
    debug_level : int
        the level of debugging sent to the terminal by JSBSim
        - 0 is limited
        - 1 is core values
        - 2 gives all calls within the C++ source code

    Methods:
    ------
    simulation_loop(profile)
        updates airsim and JSBSim in the loop
    get_graph_data()
        gets the information required to produce debug type graphics
    generate_figures()
        produce required graphics

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method docstrings have been adapted or copied from the
    OpenAI Gym source code. Methods from gym-JSBSim have also been used throughout:
    https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/environment.py
    """
    def __init__(self,
                 task_type: Type[LandingTask],
                 aircraft: Aircraft = x8,
                 agent_frequency_hz: float = 12.0,
                 graphic_frequency_hz: float = 24.0,
                 sim_frequency_hz: float = 240.0):
        if agent_frequency_hz > sim_frequency_hz:
            raise ValueError('agent frequency must be <= simulation frequency')
        self.sim: Simulation = None
        self.sim_frequency_hz = sim_frequency_hz
        self.sim_steps_per_agent_step: int = int(sim_frequency_hz // agent_frequency_hz)
        self.sim_steps_per_graphic_step: int = int(sim_frequency_hz // graphic_frequency_hz)
        self.aircraft = aircraft
        self.task = task_type(agent_frequency_hz, graphic_frequency_hz, sim_frequency_hz, aircraft)
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        self.display_graphics: bool = True

        # self.sim: Simulation = Simulation(sim_frequency_hz, aircraft, init_conditions, debug_level)
        # self.agent_interaction_frequency = agent_interaction_frequency
        # self.sim_frequency_hz = sim_frequency_hz
        # self.airsim_frequency_hz = airsim_frequency_hz
        # self.ap: X8Autopilot = X8Autopilot(self.sim)
        # self.graph: DebugGraphs = DebugGraphs(self.sim)
        # self.debug_aero: DebugFDM = DebugFDM(self.sim)
        # self.wind_estimate: WindEstimation = WindEstimation(self.sim)
        # self.over: bool = False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Runs a single step of the environment. When the end is reached reset() needs to be called to go back to the
        initial environment state.

        :param action: the agents action, with the same length as the action variables.
        :return:
            state: agent's observation of the current environment
            reward: reward generated from previous action
            done: defines whether the episode has ended and if further steps need to be called
            info: extra information
        """
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        state, reward, done, info = self.task.task_step(self.sim, action,
                                                        self.sim_steps_per_agent_step, self.sim_steps_per_graphic_step)
        return np.array(state), reward, done, info

    def reset(self) -> np.array:
        """
        Resets the state of the environment and returns an initial observation.

        :return: the initial state space observation
        """
        init_conditions = self.task.get_initial_conditions()

        if self.sim:
            self.sim.reinitialise(init_conditions)
        else:
            self.sim = self._init_new_sim(self.sim_frequency_hz, self.aircraft, init_conditions)

        state = self.task.observe_first_state(self.sim)

        return np.array(state)

    @staticmethod
    def _init_new_sim(dt, aircraft, initial_conditions) -> Simulation:
        """
        Start a new instance of the Simulation class

        :param dt: simulation frequency [Hz]
        :param aircraft: aircraft type used should normally be the x8 UAV aircraft
        :param initial_conditions: the aircraft's initial simulation conditions
        :return: a Simulation object, class instance
        """
        return Simulation(sim_frequency_hz=dt,
                          aircraft=aircraft,
                          init_conditions=initial_conditions)

    def render(self, mode: str = 'human') -> None:
        """
        Renders the environment

        From OpenAI Gyms abstract definition:
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.


        :type mode:
        :param mode:
        :return:
        """
        if mode == 'human':
            self.sim.update_airsim()
        elif mode == 'rgb_array':
            pass
        elif mode == 'ansi':
            pass
        else:
            print('valid render mode not detected, use human, rgb_array or ansi')

    def close(self):
        """
        Cleans up the environment's objects

        Environments automatically close() when garbage collected or when the program exits.
        :return:
        """
        if self.sim:
            self.sim.close()

    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return


