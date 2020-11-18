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
from Navigation import WindEstimation
from image_processing import AirSimImages, SemanticImageSegmentation
from typing import Type, Tuple, Dict


class ClosedLoop():
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
    simulation_loop(profile : tuple(tuple))
        updates airsim and JSBSim in the loop
    get_graph_data()
        gets the information required to produce debug type graphics
    generate_figures()
        produce required graphics
    """
    def __init__(self, sim_time: float,
                 display_graphics: bool = True,
                 airspeed: float = 50.0,
                 agent_interaction_frequency: float = 12.0,
                 airsim_frequency_hz: float = 24.0,
                 sim_frequency_hz: float = 240.0,
                 aircraft: Aircraft = x8,
                 init_conditions: bool = None,
                 debug_level: int = 0):
        self.sim_time = sim_time
        self.display_graphics = display_graphics
        self.airspeed = airspeed
        self.aircraft = aircraft
        self.sim: Simulation = Simulation(sim_frequency_hz, aircraft, init_conditions, debug_level)
        self.agent_interaction_frequency = agent_interaction_frequency
        self.sim_frequency_hz = sim_frequency_hz
        self.airsim_frequency_hz = airsim_frequency_hz
        self.ap: X8Autopilot = X8Autopilot(self.sim)
        self.graph: DebugGraphs = DebugGraphs(self.sim)
        self.debug_aero: DebugFDM = DebugFDM(self.sim)
        self.wind_estimate: WindEstimation = WindEstimation(self.sim)
        self.over: bool = False

    def simulation_loop(self, profile: tuple) -> None:
        """
        Runs the closed loop simulation and updates to airsim simulation based on the class level definitions

        :param profile: a tuple of tuples of the aircraft's profile in (lat [m], long [m], alt [feet])
        :return: None
        """
        update_num = int(self.sim_time * self.sim_frequency_hz)  # how many simulation steps to update the simulation
        relative_update = self.airsim_frequency_hz / self.sim_frequency_hz  # rate between airsim and JSBSim
        graphic_update = 0
        for i in range(update_num):
            graphic_i = relative_update * i
            graphic_update_old = graphic_update
            graphic_update = graphic_i // 1.0


            print(graphic_i, graphic_update_old, graphic_update)
            print(self.display_graphics)
            if self.display_graphics and graphic_update > graphic_update_old:
                self.sim.update_airsim()
                print('update_airsim')
            self.ap.airspeed_hold_w_throttle(self.airspeed)
            self.get_graph_data()
            if not self.over:
                self.over = self.ap.arc_path(profile, 200)
            if self.over:
                print('over and out!')
                break
            self.sim.run()

    def get_graph_data(self) -> None:
        """
        Gets the information required to produce debug type graphics

        :return:
        """
        self.graph.get_abs_pos_data()
        self.graph.get_airspeed()
        self.graph.get_alpha()
        self.graph.get_control_data()
        self.graph.get_time_data()
        self.graph.get_pos_data()
        self.graph.get_angle_data()

    def generate_figures(self) -> None:
        """
        Produce required graphics, outputs them in the desired graphic environment

        :return: None
        """
        self.graph.control_plot()
        self.graph.trace_plot_abs()
        self.graph.three_d_scene()


def run_simulator() -> None:
    """
    Runs the JSBSim and Airsim in the loop when executed as a script

    :return: None
    """
    env = ClosedLoop(100, True)
    # circuit_profile = ((0, 0, 1000), (4000, 0, 1000), (4000, 4000, 1000), (0, 4000, 1000), (0, 0, 20),
    #                    (4000, 0, 20), (4000, 4000, 20))
    ice_profile = ((0, 0, 0), (1200, 0, 0), (1300, 150, 0), (540, 530, -80), (0, 0, -150), (100, 100, -100))
    env.simulation_loop(ice_profile)
    env.generate_figures()
    print('Simulation ended')


if __name__ == '__main__':
    run_simulator()



