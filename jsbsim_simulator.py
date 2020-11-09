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
        """
        Load a JSBSim xml formatted aircraft model into the JSBSim flight dynamic model

        :param model_name: name of aircraft model loaded into JSBSim
        :return: None
        """
        load_success = self.fdm.load_model(model_name)

        if not load_success:
            raise RuntimeError('JSBSim could not find specified model name: ' + model_name)

    def get_aircraft(self) -> Aircraft:
        """
        Get the Aircraft the JSBSim was initialised with

        :return: aircraft used in the simulator
        """
        return self.aircraft

    def get_loaded_model_name(self) -> str:
        """
        Get the name of the loaded aircraft model from the current JSBSim FDM instance

        :return: JSBSim model name
        """
        name: str = self.fdm.get_model_name().decode(self.encoding)
        if name:
            return name
        else:
            return None

    def initialise(self, dt: float, model_name: str, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Start JSBSim with custom initial conditions

        :param dt: simulation rate [s]
        :param model_name: the aircraft model used
        :param init_conditions: initial simulation conditions
        :return: None
        """
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

    def set_custom_initial_conditions(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Set initial conditions different to what is found in the <name-ic.xml> file used

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties
        :return: None
        """
        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self[prop] = value

    def reinitialise(self, init_conditions: Dict['prp.Property', float] = None) -> None:
        """
        Restart the simulator with initial conditions

        :param init_conditions: the initial simulation conditions, defined based on prp JSBSim properties,
        by default this is the original initialization file
        :return: None
        """
        self.set_custom_initial_conditions(init_conditions=init_conditions)
        no_output_reset_mode = 0
        self.fdm.reset_to_initial_conditions(no_output_reset_mode)

    def run(self) -> bool:
        """
        Check if the FDM has terminated and if not advances one time step, slows by wall_clock_dt

        :return: True if FDM can advance
        """
        result = self.fdm.run()
        if self.wall_clock_dt is not None:
            time.sleep(self.wall_clock_dt)
        return result

    def get_time(self) -> float:
        """
        Get the current simulation time

        :return: the simulation time
        """
        sim_time = self[prp.sim_time_s]
        return sim_time

    def get_local_position(self) -> list:
        """
        Get the local absolute position from the simulation start point

        :return: position [lat, long, alt]
        """
        lat = self[prp.lng_travel_m]
        long = self[prp.lat_travel_m]
        alt = self[prp.altitude_sl_ft]
        position = [lat, long, alt]
        return position

    def get_local_orientation(self):
        """
        Get the orientation of the vehicle

        :return: orientation [pitch, roll, yaw]
        """
        pitch = self[prp.pitch_rad]
        roll = self[prp.roll_rad]
        yaw = self[prp.heading_deg] * (math.pi / 180)
        orientation = [pitch, roll, yaw]
        return orientation

    @staticmethod
    def airsim_connect() -> airsim.VehicleClient:
        """
        Connect to airsim client, exposing the CV mode UE4 graphic environment.

        :return: the airsim client object
        """
        client = airsim.VehicleClient()
        client.confirmConnection()
        return client

    def update_airsim(self) -> None:
        """
        Update airsim with vehicle pose calculated by JSBSim

        :return: None
        """
        pose = self.client.simGetVehiclePose()
        position = self.get_local_position()
        pose.position.x_val = position[0]
        pose.position.y_val = position[1]
        pose.position.z_val = - position[2]
        euler_angles = self.get_local_orientation()
        pose.orientation = airsim.to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
        self.client.simSetVehiclePose(pose, True)

    def close(self) -> None:
        """
        Close the JSBSim Flight Dynamic Model (FDM) currently running

        :return: None
        """
        if self.fdm:
            self.fdm = None

    def start_engines(self) -> None:
        """
        Start all available aircraft propulsion units

        :return: None
        """
        self[prp.all_engine_running] = -1

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float) -> None:
        """
        Set the throttle and mixture propulsion commands on an ICE powerplant, allows for a 2 engine aircraft too

        :param throttle_cmd: controls the throttle deflection (0 <-> 1)
        :param mixture_cmd: controls the mixture deflection (0 <-> 1)
        :return:
        """
        self[prp.throttle_cmd] = throttle_cmd
        self[prp.mixture_cmd] = mixture_cmd

        try:
            self[prp.throttle_1_cmd] = throttle_cmd
            self[prp.mixture_1_cmd] = mixture_cmd
        except KeyError:
            pass  # must be single-control aircraft

    def raise_landing_gear(self) -> None:
        """
        Raise the aircraft's landing gear

        :return: None
        """
        self[prp.gear] = 0.0
        self[prp.gear_all_cmd] = 0.0
