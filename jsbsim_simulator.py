import jsbsim
import airsim
import os
import time
from typing import Dict, Union
import jsbsim_properties as prp
from jsbsim_aircraft import Aircraft, cessna172P
import math

"""Initially based upon https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/simulation.py by Gordon Rennie"""


class Simulation:
    """
    The core JSBSim simulation class

    ...

    Attributes
    ----------
    fdm : object
        an object that is an instance of the JSBSim's flight dynamic model
    sim_dt : var
        the simulation update rate
    aircraft : Aircraft
        the aircraft type used, cessna172P by default
    init_conditions : float
        the simulations initial conditions None by default as in basic_ic.xml
    debug_level : int
        the level of debugging sent to the terminal by jsbsim
        - 0 is limited
        - 1 is core values
        - 2 gives all calls within the C++ source code
    wall_clock_dt : bool
        activates a switch to speed up or slow down the simulation
    client : object
        connection to airsim for visualization

    Methods
    ------
    load_model(model_name)
        Ensure the JSBSim flight dynamic model is found and loaded in correctly
    get_aircraft()
        returns the aircraft the simulator was initialized with
    get_loaded_model_name()
        returns the name of the fdm model used
    initialise(dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None)
        initializes an instance of JSBSim
    set_custom_initial_conditions(init_conditions: Dict['prp.Property', float] = None)
        allows for initial conditions different to basic_ic.xml to be used
    reinitialise(self, init_conditions: Dict['prp.Property', float] = None)
        restart the simulation with default initial conditions
    run()
        run JSBSim at the sim_dt rate
    get_time()
        returns the current JSBSim time
    get_local_position()
        returns the lat, long and altitude of JSBSim
    get_local_orientation()
        returns the euler angle orientation (roll, pitch, yaw) of JSBSim
    airsim_connect()
        connect to a running instance of airsim
    update_airsim()
        updates the airsim client with the JSBSim calculated pose information
    close()
        closes the JSBSim fdm instance
    start_engines()
        starts all available aircraft engines
    set_throttle_mixture_controls()
        sets aircraft mixture and throttle controls
    raise_landing_gear()
        raises the aircraft's landing gear
    """


    encoding = 'utf-8'
    ROOT_DIR = os.path.abspath('/home/quessy/Dev/jsbsim')

    def __init__(self,
                 sim_frequency_hz: float = 60.0,
                 aircraft: Aircraft = cessna172P,
                 init_conditions: Dict[prp.Property, float] = None,
                 debug_level: int = 0):
        self.fdm = jsbsim.FGFDMExec(root_dir=self.ROOT_DIR)
        self.fdm.set_debug_level(debug_level)
        self.sim_dt = 1.0 / sim_frequency_hz
        self.aircraft = aircraft
        self.initialise(self.sim_dt, self.aircraft.jsbsim_id, init_conditions)
        self.fdm.disable_output()
        self.wall_clock_dt = None
        self.client = self.airsim_connect()

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        return self.fdm[prop.name]

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        self.fdm[prop.name] = value

    def load_model(self, model_name: str) -> None:
        load_success = self.fdm.load_model(model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model name: ' + model_name)

    def get_aircraft(self) -> Aircraft:
        """
        Gets the Aircraft the sim was initialised with.
        """
        return self.aircraft

    def get_loaded_model_name(self) -> str:
        name: str = self.fdm.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            return None

    def initialise(self, dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None) -> None:
        if init_conditions is not None:
            ic_file = 'minimal_ic.xml'
        else:
            ic_file = 'basic_ic.xml'

        ic_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ic_file)
        self.fdm.load_ic(ic_path, useStoredPath=False)
        self.load_model(model_name)
        self.fdm.set_dt(dt)
        self.set_custom_initial_conditions(init_conditions)

        success = self.fdm.run_ic()
        if not success:
            raise RuntimeError('JSBSim failed to initialise simulation conditions.')

    def set_custom_initial_conditions(self,
                                      init_conditions: Dict['prp.Property', float] = None) -> None:
        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self[prop] = value

    def reinitialise(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        self.set_custom_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 0
        self.fdm.reset_to_initial_conditions(no_output_reset_mode)

    def run(self) -> bool:
        result = self.fdm.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def get_time(self):
        sim_time = self[prp.sim_time_s]
        return sim_time

    def get_local_position(self):
        lat = self[prp.lng_travel_m]
        long = self[prp.lat_travel_m]
        alt = self[prp.altitude_sl_ft]
        position = [lat, long, alt]
        return position

    def get_local_orientation(self):
        pitch = self[prp.pitch_rad]
        roll = self[prp.roll_rad]
        yaw = self[prp.heading_deg] * (math.pi / 180)
        orientation = [pitch, roll, yaw]
        return orientation

    @staticmethod
    def airsim_connect():
        client = airsim.VehicleClient()
        client.confirmConnection()
        return client

    def update_airsim(self):
        pose = self.client.simGetVehiclePose()
        position = self.get_local_position()
        pose.position.x_val = position[0]
        pose.position.y_val = position[1]
        pose.position.z_val = - position[2]
        euler_angles = self.get_local_orientation()
        pose.orientation = airsim.to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
        self.client.simSetVehiclePose(pose, True)

    def close(self):
        if self.fdm:
            self.fdm = None

    def start_engines(self):
        self[prp.all_engine_running] = -1

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float):
        self[prp.throttle_cmd] = throttle_cmd
        self[prp.mixture_cmd] = mixture_cmd

        try:
            self[prp.throttle_1_cmd] = throttle_cmd
            self[prp.mixture_1_cmd] = mixture_cmd
        except KeyError:
            pass  # must be single-control aircraft

    def raise_landing_gear(self):
        self[prp.gear] = 0.0
        self[prp.gear_all_cmd] = 0.0
