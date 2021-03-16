import airsim
from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from debug_utils import *
import jsbsim_properties as prp
from autopilot import X8Autopilot
from image_processing import AirSimImages
from typing import Dict
from report_diagrams import ReportGraphs

class ImagePath:
    """
    A class to run airsim for the purpose of obtaining images from altitude

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
    """
    def __init__(self, sim_time: float,
                 display_graphics: bool = True,
                 airspeed: float = 50.0,
                 airsim_frequency_hz: float = 24.0,
                 agent_interaction_frequency: float = 12.0,
                 sim_frequency_hz: float = 240.0,
                 aircraft: Aircraft = x8,
                 init_conditions: Dict[prp.Property, float] = None,
                 debug_level: int = 0
                 ):
        self.sim_time = sim_time
        self.display_graphics = display_graphics
        self.airspeed = airspeed
        self.aircraft = aircraft
        self.sim: Simulation = Simulation(sim_frequency_hz, aircraft, init_conditions, debug_level)
        self.agent_interaction_frequency = agent_interaction_frequency
        self.sim_frequency_hz = sim_frequency_hz
        self.airsim_frequency_hz = airsim_frequency_hz
        self.ap: X8Autopilot = X8Autopilot(self.sim)
        #  self.report: ReportGraphs = ReportGraphs(self.sim)
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
        image = AirSimImages(self.sim)
        image.get_np_image(image_type=airsim.ImageType.Scene)
        for i in range(update_num):
            print(self.sim[prp.altitude_sl_ft])
            graphic_i = relative_update * i
            graphic_update_old = graphic_update
            graphic_update = graphic_i // 1.0
            if self.display_graphics and graphic_update > graphic_update_old:
                self.sim.update_airsim()
            self.ap.airspeed_hold_w_throttle(self.airspeed)
            if not self.over:
                self.over = self.ap.arc_path(profile, 400)
            if self.over:
                print('over and out!')
                break
            self.sim.run()


def simulate() -> None:
    """Runs the JSBSim and AirSim in the loop when executed as a script

    :return: None
    """

    runway_start = {
        prp.initial_altitude_ft: 100,
        prp.initial_longitude_geoc_deg: 0.0 / 111120.0,
        prp.initial_latitude_geod_deg: 0.0 / 111120.0,
        prp.initial_u_fps: 50.0,
        prp.initial_w_fps: 0.0,
        prp.initial_heading_deg: 30.0,
        prp.initial_roc_fpm: 0.0,
        prp.all_engine_running: -1
    }
    env = ImagePath(sim_time=750, display_graphics=True, init_conditions=runway_start, airsim_frequency_hz=24)
    rectangle = ((0, 0, 0), (2000, 0, 100), (2000, 2000, 100), (-2000, 2000, 100), (-2000, 0, 100), (2000, 0, 20),
                 (2000, 2000, 20), (-2000, 2000, 20))
    # angle = 30.0
    # straight_line = [[0, 0, 0], [2001, 0, 100], [2000, 2000, 100], [-2000, 2000, 100], [-2000, 0, 20], [2000, 0, 20]]
    # idx = 0
    # for point in straight_line:
    #     x = point[0] * math.sin(angle * (math.pi / 180.0))
    #     y = point[1] * math.cos(angle * (math.pi / 180.0))
    #     straight_line[idx] = tuple([x, y, straight_line[idx][2]])
    #     idx += 1
    # straight_line = tuple(straight_line)
    env.simulation_loop(rectangle)
    print("Simulation Ended")


if __name__ == '__main__':
    simulate()
