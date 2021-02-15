import gym
import enum
import types
import math
import enum
import warnings
import numpy as np
import jsbsim_properties as prp
from abc import ABC, abstractmethod
from collections import namedtuple
from jsbsim_simulator import Simulation
from jsbsim_properties import BoundedProperty, Property
from jsbsim_aircraft import Aircraft
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type

"""ATTRIBUTION: Abstract Task Class based on -> https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/tasks.py"""


class Task(ABC):
    """
    Interface for each Task type in the JSBSim/Airsim environment. A task, defines its own:
        - state space
        - action space
        - termination condition
        - agent_reward function

        ...

        Attributes:
        ----------

        Methods:
        -------
        task_step(sim, action, sim_steps_per_agent_step, sim_steps_per_graphic_step)
            calculates the new state, reward and termination condition
        observe_first_state(sim)
            initialize any state/controls and get first state observation from reset sim
        get_initial_conditions()
            dictionary mapping from initial episode conditions to values
        get_state_space()
            get the task's state space object
        get_action_space
            get the task's action space object

    """

    @abstractmethod
    def task_step(self, sim: Simulation, action: Sequence[float],
                  sim_steps_per_agent_step: float, sim_steps_per_graphic_step: float) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Calculates the new state, reward and termination condition

        :param sim:
            an instance of the Simulation class to get the agent's state from
        :param action:
            a sequence of floats, the agent's last action
        :param sim_steps_per_agent_step:
            the number of JSBSim steps to perform following an action to make another observation
        :param sim_steps_per_graphic_step:
            the number of JSBSim steps to perform following a graphic update
        :return: tuple of (observation, reward, done, info) where:
            - observation, is an array of the agent's observation of the environment state
            - reward, the reward for the action performed
            - done, True if the episode is over, otherwise it is False
            - info, optional dict containing useful information for debugging
        """

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        """
        Initialize any state/controls and get first state observation from reset sim

        :param sim: Simulation, the environment simulation
        :return: np array the first state observation of the episode
        """

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        """
        Dictionary mapping from initial episode conditions to values

        ....

        Episode initial conditions (ICs) are defined by specifying values for
        JSBSim properties, represented by their name (string) in JSBSim.

        JSBSim uses a distinct set of properties for ICs, beginning with 'ic/'
        which differ from property names during the simulation, e.g. "ic/u-fps"
        instead of "velocities/u-fps". See https://jsbsim-team.github.io/jsbsim/

        :return: dict mapping string for each initial condition property to
            initial value, a float, or None to use Env defaults
        """

    @abstractmethod
    def get_state_space(self) -> gym.Space:
        """
        Get the task's state space object
        """

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        """
        Get the task's action space object
        """


class FlightTask(Task, ABC):
    """
    Abstract Superclass for flight tasks

    ...

    Attributes:
    ----------
    state_variables: tuple
        the task's state representation
    action_variables: tuple
        the task's actions

    Methods:
    -------
    get_initial_conditions()
        dict mapping InitialProperties to initial values
    _is_terminal()
        determines episode termination
    _new_episode_init() (optional)
        performs any control input/initialisation on episode reset
    _update_custom_properties (optional)
        updates any custom properties in the sim
    """

    INITIAL_ALTITUDE_FT = 2000
    # State variables provided by default to agent
    base_state_variables = (prp.altitude_sl_ft,  # altimeter
                            prp.pitch_rad, prp.roll_rad,  # AI
                            prp.heading_deg,  # DI
                            prp.lat_geod_deg, prp.lng_geoc_deg,  # GNSS receiver
                            prp.airspeed,  # ASI
                            prp.w_fps,  # VSI
                            prp.aileron_combined_rad, prp.elevator_rad  # CC position
                            )
    base_initial_conditions = types.MappingProxyType(
        {
            prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
            prp.initial_terrain_altitude_ft: 0.00000001,
            prp.initial_longitude_geoc_deg: 0.0,
            prp.initial_latitude_geod_deg: 0.0
        }
    )

    state_variables: Tuple[BoundedProperty, ...]
    action_variables: Tuple[BoundedProperty, ...]
    State: Type[NamedTuple]

    def __init__(self, debug: bool = False) -> None:
        self.last_state = None
        self._make_state_class()
        self.debug = debug

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    @abstractmethod
    def _is_terminal(self, sim: Simulation) -> bool:
        """
        Determines whether the current episode should terminate.

        :param sim: the current simulation
        :return: True if the episode should terminate else False
        """
        ...

    @abstractmethod
    def set_sequential_reward(self) -> float:
        """
        Set the reward generated at each task_step

        :return: reward
        """
        ...

    @abstractmethod
    def set_terminal_reward(self, sim: Simulation) -> float:
        """
        Set the reward generated at the end of the episode

        :return: reward
        """
        ...

    def _make_state_class(self) -> None:
        """
        Creates a namedtuple for readable state data

        :return: None
        """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float],
                  sim_steps_per_agent_step: int, sim_steps_per_graphic_step: int, display_graphics: bool = True) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Advances the task specific gym step to determine calculate key MDP variables for the RL agent

        :param sim: core aircraft simulation
        :param action: actions taken for the next timestep
        :param sim_steps_per_agent_step: number of steps jsbsim takes per observation
        :param sim_steps_per_graphic_step: number of steps jsbsim takes per airsim step
        :param display_graphics: whether or not to run airsim graphics out, defaults to being on
        :return:
            state: agent's observation of the current environment
                reward: reward generated from previous action
                done: defines whether the episode has ended and if further steps need to be called
                info: extra information
        """

        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        graphic_update = 0
        # run jsbsim and airsim until another observation is required
        relative_update = 1 / sim_steps_per_graphic_step
        for i in range(sim_steps_per_agent_step):
            graphic_i = relative_update * i
            graphic_update_old = graphic_update
            graphic_update = int(graphic_i // 1.0)
            if display_graphics and graphic_update > graphic_update_old:
                sim.update_airsim()
            sim.run()

        state = self.State(*(sim[prop] for prop in self.state_variables))
        done = self._is_terminal(sim)
        reward = self.set_sequential_reward()  # step-wise reward value, should be mapped from a reward function
        # reward = self.assessor.assess(state, self.last_state, done)
        if done:
            reward = self.set_terminal_reward(sim)  # should be episodic reward
        self._store_reward(reward, sim)
        self.last_state = state  # update state
        info = {'reward': reward}
        return state, reward, done, info

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._init_new_episode(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        return state

    def _init_new_episode(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        This is currently empty null but should set the reward to 0.0
        """
        # insert method to set reward to 0

    def _store_reward(self, reward: float):
        """
        Store the agent's total reward, this is the return G = sum(r_t)
        :param reward:
        :return:
        """

    def get_state_space(self) -> gym.Space:
        """
        Samples state space highs and lows

        :return: gym.Space with the lowest and highest state variables within the space
        """
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        """
        Samples action space highs and lows

        :return: gym.Space with the lowest and highest action variables within the space
        """
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')


class LandingTask(FlightTask):
    """
    A task where the agent should select and land at a suitable landing site

    ...

    Attributes:
    ---------
    aircraft: Aircraft
        type of aircraft used in the task, can override this here to another type if desired

    Methods:
    -------
    _is_terminal(sim)
        overrides the abstract method in FlightTask with the terminating conditions, terminates the episode
        when the aircraft collides with the ground
    set_sequential_reward()
        calculate the sequential reward obtained by the agent throughout the episode before termination
    set_terminal_reward(sim)
        Calculate the reward at the end of an episode
    _init_new_episode()
        begin a new episode with any task specific attributes different to the generic flight task init_new_episode


    """

    # Initially 3 possible class level actions are defined airspeed, altitude and heading
    target_airspeed = BoundedProperty('target/airspeed', 'desired airspeed [ktas]', 0.0, 100.0)
    target_altitude = BoundedProperty('target/altitude', 'desired altitude [feet]',
                                      prp.altitude_sl_ft.min, prp.altitude_sl_ft.max)
    target_heading = BoundedProperty('target/heading', 'desired heading [deg]',
                                     rp.heading_deg.min, prp.heading_deg.max)

    action_variables = (target_airspeed, target_altitude, target_heading) 

    def __init__(self,
                 aircraft: Aircraft):

        self.aircraft = aircraft
        self.state_variables = FlightTask.base_state_variables
        super().__init__()

    def _is_terminal(self, sim: Simulation) -> bool:
        """
        Overrides the abstract method in FlightTask with the terminating conditions, terminates the episode
        when the aircraft collides with the ground

        :param sim: JSBSim/Airsim instance
        :return: true if the episode is finished/done
        """
        collision = sim.get_collision_info()
        return collision

    def set_sequential_reward(self) -> float:
        """
        Calculate the sequential reward obtained by the agent throughout the episode before termination

        Currently sequential reward is 0, for the engine off glide this will be fine but if landing without power may
        cause the agent to wonder forever. To minimise time set this value to 0.

        :return: reward
        """
        reward = 0.0
        return reward

    def set_terminal_reward(self, sim: Simulation) -> float:
        """
        Calculate the reward at the end of an episode

        :param sim:
        :return:
        """
        if self._crash_test(sim):
            reward = -1
            return reward
        elif self._restricted_area(sim, restricted_objects):
            reward = -1
            return reward
        elif self._safe_zone(sim, safe_objects):
            reward = self.safe_landing_reward(sim)
            return reward

    def safe_landing_reward(self, sim: Simulation) -> float:
        """
        Calculate the reward generated when landing in a sensible location

        :param sim:
        :return:
        """
        airspeed_reward = self.min_component(sim[prp.airspeed], 0.1)
        rod_reward = self.min_component(sim[prp.w_fps], 0.5)
        lda_reward = self.max_component(self.landing_distance, 1.0)
        reward_values = [airspeed_reward, rod_reward, lda_reward]
        airspeed_weight = 2.0
        rod_weight = 1.0
        lda_weight = 0.2
        reward_weights = [airspeed_weight, rod_weight, lda_weight]
        reward = self.weighted_average(reward_values, reward_weights)
        return reward

    @staticmethod
    def weighted_average(values: [float, ...], weights: [float, ...]) -> float:
        """
        Calculates the weighted average of a group of values

        :param values:
        :param weights:
        :return: weight averaged values
        """
        if len(values) == len(weights):
            weighted_value = 0
            for i in values:
                weighted_value = weighted_value + (values[i] * weights[i])
            weighted_average = weighted_value / len(values)
            return weighted_average
        else:
            ValueError(f'length of values (', len(values), f') and weights (', len(weights), f') must be the same')

    @staticmethod
    def max_component(assessment_value: float, reward_sharpness: float) -> float:
        """
        Generate a reward that aims to maximise a given assessment_value

        :param assessment_value: the value being assessed
        :param reward_sharpness: the rate of reward decay away from the maximum:
            - a low value is a very sharp reward function
            - a high value gives a shallow reward function
        :return: a reward value in range [0, 1]
        """
        if reward_sharpness < 0:
            state_scaling = abs(assessment_value) / reward_sharpness
            reward = (state_scaling / (1 + state_scaling))
            return reward
        else:
            ValueError('reward_sharpness must be > 0')

    @staticmethod
    def min_component(assessment_value: float, reward_sharpness: float) -> float:
        """
        Generate a reward that aims to maximise a given assessment_value

        :param assessment_value: the value being assessed
        :param reward_sharpness: the rate of reward decay away from the maximum:
            - a low value is a very sharp reward function
            - a high value gives a shallow reward function
        :return: a reward value between [0, 1]
        """
        if reward_sharpness < 0:
            state_scaling = abs(assessment_value) / reward_sharpness
            reward = 1 - (state_scaling / (1 + state_scaling))
            return reward
        else:
            ValueError('reward_sharpness must be > 0')

    def _init_new_episode(self, sim: Simulation):
        super()._init_new_episode(sim)

    @staticmethod
    def _crash_test(sim: Simulation) -> bool:
        """
        Check if the aircraft crashed by landing out of limits

        ...

        Limits imposed on the terminal crash condition are as follows:
            - -20 degs < roll < +20 degs
            - -5 degs < pitch < +10 degs
            - 15 m/s < RoD
        :param sim: Simulation object instance
        :return: true if aircraft crashed
        """
        if -0.35 > sim[prp.roll_rad] > 0.35 \
                or -0.09 > sim[prp.pitch_rad] > 0.18 \
                or sim[prp.w_fps] > 50:
            crash = True
        else:
            crash = False
        return crash

    @staticmethod
    def _restricted_area(sim: Simulation, restricted_objects: tuple) -> bool:
        """
        Check if the aircraft landed within a restricted zone

        :param sim: simulation object instance
        :param restricted_objects: a tuple of the UE4 environment simulation objects that the agent should not land in
        :return: true if aircraft landed in restricted zone
        """
        collision = sim.get_collision_info()
        collision_object = collision.object_id
        restricted_area = False
        if collision_object in restricted_objects:
            restricted_area = True
        return restricted_area

    @staticmethod
    def _safe_zone(sim: Simulation, safe_objects: tuple) -> bool:
        """
        Check if the aircraft landed within a safe zone

        :param sim: simulation object instance
        :param safe_objects: a tuple of the UE4 environment simulation objects that the agent should land in
        :return: true if aircraft landed in safe zone
        """
        collision = sim.get_collision_info()
        collision_object = collision.object_id
        safe_area = False
        if collision_object in safe_objects:
            safe_area = True
        return safe_area


