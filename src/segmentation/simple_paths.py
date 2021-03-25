import airsim
from src.jsbsim_simulator import Simulation
from src.jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from src.debug_utils import *
import src.navigation as navigation
import src.jsbsim_properties as prp
from src.autopilot import X8Autopilot
from src.image_processing import AirSimImages
from typing import Dict
from src.report_diagrams import ReportGraphs
import os
import numpy as np
import cv2
import time
import json


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
                 airspeed: float = 30.0,
                 agent_interaction_frequency: float = 12.0,
                 airsim_frequency_hz: float = 24.0,
                 sim_frequency_hz: float = 240.0,
                 aircraft: Aircraft = x8,
                 init_conditions: Dict['prp.Property', float] = None,
                 debug_level: int = 0,
                 dataset_name: str = 'default_dataset'):
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
        self.report: ReportGraphs = ReportGraphs(self.sim)
        self.debug_aero: DebugFDM = DebugFDM(self.sim)
        self.over: bool = False
        self.dataset_name = dataset_name

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
        self.setup_semantic_segmentation()
        self.setup_dataset_directories(self.dataset_name)
        seg_id, image_id = 0, 0
        for i in range(update_num):
            graphic_i = relative_update * i
            graphic_update_old = graphic_update
            graphic_update = graphic_i // 1.0
            if self.display_graphics and graphic_update > graphic_update_old:
                self.sim.update_airsim()
                seg_id, image_id = self.store_image_set(self.dataset_name, seg_id, image_id)
                self.store_flight_data(self.dataset_name, image_id)
            self.ap.airspeed_hold_w_throttle(self.airspeed)
            if not self.over:
                self.over = self.ap.arc_path(profile, 200)
            if self.over:
                print('over and out!')
                break
            self.get_graph_data()
            self.sim.run()

    @staticmethod
    def rotate_vector(vec: list, angle: float) -> list:
        """
        Rotates a 2D vector by a given angle

        :param vec: 2 vector points to be rotated the [0, 0] origin
        :param angle: rotation angle in degrees
        :return: 2 points mapping the rotated vector
        """
        x = (vec[0] * math.cos(angle * (math.pi / 180.0))) - (vec[1] * math.sin(angle * (math.pi / 180.0)))
        y = (vec[1] * math.cos(angle * (math.pi / 180.0))) + (vec[0] * math.sin(angle * (math.pi / 180.0)))
        return [x, y]

    def transform_path(self, vec: list, angle: float, origin: list) -> tuple:
        """
        Transforms a defined vector shape into another vector

        :param vec: set of points [x, y, alt] to fly from the origin
        :param angle: the angle to rotate the vector shape by
        :param origin: the [0,0] point to set the shape from (usually initialization)
        :return: points for A/P
        """
        idx = 0
        for point in vec:
            #  Using rotation matrix to rotate vectors of points
            rot_vec = self.rotate_vector([point[0], point[1]], angle)
            vec[idx] = [rot_vec[0] + (origin[0]), rot_vec[1] + (origin[1]), vec[idx][2]]
            idx += 1
        return tuple(vec)

    def get_graph_data(self) -> None:
        """
        Retrieves the information required to produce debug type graphics

        :return:
        """
        self.graph.get_abs_pos_data()
        self.graph.get_airspeed()
        self.graph.get_alpha()
        self.graph.get_control_data()
        self.graph.get_time_data()
        self.graph.get_pos_data()
        self.graph.get_angle_data()
        self.graph.get_rate_data()
        self.report.get_graph_info()

    def generate_figures(self) -> None:
        """
        Produce required graphics, outputs them in the desired graphic environment

        :return: None
        """
        self.graph.trace_plot_abs()

    def setup_semantic_segmentation(self):
        """
        Setup the patches of semantic segmentation based on static_meshes within the image

        :return: unique colours of each image
        """
        # maps segmentationID to image rgb code
        seg_vals = {
            0: (0, 0, 0),
            20: (15, 8, 73),
            21: (70, 76, 194),
            22: (104, 53, 78),
            23: (0, 5, 137),
            24: (90, 47, 155)
        }
        # maps labels to index
        colour_labels = {
            "building": 21,
            "runway": 22,
            "sky": 0,
            "ground": 24
        }
        found_cafe = self.sim.client.simSetSegmentationObjectID("cafe_building_17", 24, True)
        found_hangar_1 = self.sim.client.simSetSegmentationObjectID("Hangar_1", 24, True)
        found_hangar_2 = self.sim.client.simSetSegmentationObjectID("Hangar_2", 24, True)
        found_hangar_3 = self.sim.client.simSetSegmentationObjectID("Hangar_3", 24, True)
        found_hangar_4 = self.sim.client.simSetSegmentationObjectID("Hangar_4", 24, True)
        found_runway = self.sim.client.simSetSegmentationObjectID("runway", 22, True)
        found_tower = self.sim.client.simSetSegmentationObjectID("Tower_22", 24, True)
        found_landscape = self.sim.client.simSetSegmentationObjectID("lydd_landscape_3", 24, True)

    @staticmethod
    def setup_dataset_directories(dataset: str) -> None:
        """
        Setsup the directory structure to store a dataset

        :param dataset:
        :return: None
        """
        dirname = os.path.dirname(__file__)  # get the location of the root directory
        dirname = os.path.join(dirname, '../..')  # move out of segmentation source directory
        dirname = os.path.join(dirname, 'data/segmentation-datasets')  # go into segmentation-dataset dir
        if not os.path.isdir(dirname + '/' + dataset):
            os.mkdir(dirname + '/' + dataset)
        if not os.path.isdir(dirname + '/' + dataset + '/metadata'):
            os.mkdir(dirname + '/' + dataset + '/metadata')
        if not os.path.isdir(dirname + '/' + dataset + '/flight_data'):
            os.mkdir(dirname + '/' + dataset + '/flight_data')
        if not os.path.isdir(dirname + '/' + dataset + '/images'):
            os.mkdir(dirname + '/' + dataset + '/images')
        if not os.path.isdir(dirname + '/' + dataset + '/segmentation_masks'):
            os.mkdir(dirname + '/' + dataset + '/segmentation_masks')

    def store_image_set(self, dataset: str, seg_id: int, image_id: int):
        """
        Stores a semantic image mask with a regular image as a png

        :dataset: the name of the directory where the dataset is to be stored
        :seg_id: the id of the segmentation mask
        :image_id: the id of the image
        :return: the int id used on the images and segmentation masks
        """
        dirname = os.path.dirname(__file__)  # get the location of the root directory
        dirname = os.path.join(dirname, '../..')  # move out of segmentation source directory
        dirname = os.path.join(dirname, 'data/segmentation-datasets')  # go into segmentation-dataset dir
        dirname = os.path.join(dirname, dataset)  # go into segmentation specific dataset
        seg_path = os.path.join(dirname, 'segmentation_masks')  # make a path to segmentation_masks dir specifically
        image_path = os.path.join(dirname, 'images')  # make a path to images dir specifically

        responses_seg = self.sim.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, True),  # depth in perspective projection
            airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
        ])
        for idx, response in enumerate(responses_seg):
            if len(response.image_data_uint8) != 0:
                # print(f"{response.width}x{response.height}")
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array
                cur_seg = os.path.join(seg_path, str(seg_id) + ".png")
                while os.path.exists(cur_seg):
                    seg_id += 1
                    cur_seg = os.path.join(seg_path, str(seg_id) + ".png")
                    # TODO: seg method just skips forward one would be better go to end of images
                cv2.imwrite(os.path.join(seg_path, str(seg_id) + ".png"), img_rgb)
        responses_image = self.sim.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
        for idx, response in enumerate(responses_image):
            if len(response.image_data_uint8) != 0:
                img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)  # get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3)  # reshape array to 3 channel image array
                cur_image = os.path.join(image_path, str(image_id) + ".png")
                while os.path.exists(cur_image):
                    image_id += 1
                    cur_image = os.path.join(image_path, str(image_id) + ".png")
                    # TODO: image method just skips forward one would be better go to end of images
                cv2.imwrite(os.path.join(image_path, str(image_id) + ".png"), img_rgb)
        return seg_id, image_id


    def store_flight_data(self, dataset: str, flight_id: int) -> None:
        """
        Store flight data from the simulation as a JSON object

        :dataset: the name of the directory where the dataset is to be stored
        :flight_data_id: the index of the JSON object as related to the image.
        :return: None
        """
        dirname = os.path.dirname(__file__)  # get the location of the root directory
        dirname = os.path.join(dirname, '../..')  # move out of segmentation source directory
        dirname = os.path.join(dirname, 'data/segmentation-datasets')  # go into segmentation-dataset dir
        dirname = os.path.join(dirname, dataset)  # go into segmentation specific dataset
        flight_path = os.path.join(dirname, 'flight_data')  # make a path to flight_data dir specifically

        self.nav = navigation.LocalNavigation(self.sim)
        flight_dict = {
            'image_id': flight_id,
            'time': self.sim.get_time(),
            'lat_m': self.nav.get_local_pos()[0],
            'long_m': self.nav.get_local_pos()[1],
            'alt': self.sim[prp.altitude_sl_ft],
            'pitch': self.sim.get_local_orientation()[0],
            'roll': self.sim.get_local_orientation()[1],
            'yaw': self.sim.get_local_orientation()[2],
            'u': self.sim[prp.u_fps] * 0.3048,
            'v': self.sim[prp.v_fps] * 0.3048,
            'w': self.sim[prp.w_fps] * 0.3048,
            'p': self.sim[prp.p_radps],
            'q': self.sim[prp.q_radps],
            'r': self.sim[prp.r_radps],
            'airspeed': self.sim[prp.airspeed],
            'alpha': self.sim[prp.alpha],
            'aileron_combined': self.sim[prp.aileron_combined_rad],
            'elevator': self.sim[prp.elevator_rad],
            'throttle': self.sim[prp.throttle]
        }
        json_file_name = os.path.join(flight_path, str(flight_id) + ".json")
        with open(json_file_name, 'w') as json_file:
            json.dump(flight_dict, json_file)

        
def simulate() -> None:
    """Runs the JSBSim and AirSim in the loop when executed as a script

    :return: None
    """

    runway_start = {
        prp.initial_altitude_ft: 100,
        prp.initial_latitude_geod_deg: -1000.0000000000002 / 111120.0,
        prp.initial_longitude_geoc_deg: 1732.0508075688772 / 111120.0,
        prp.initial_u_fps: 50.0,
        prp.initial_w_fps: 0.0,
        prp.initial_heading_deg: 300.0,
        prp.initial_roc_fpm: 0.0,
        prp.all_engine_running: -1
    }

    env = ImagePath(sim_time=750, display_graphics=True, init_conditions=runway_start, airsim_frequency_hz=0.2,
                    dataset_name="fd-test")
    rectangle = ((0, 0, 0), (2000, 0, 100), (2000, 2000, 100), (-2000, 2000, 100), (-2000, 0, 100), (2000, 0, 20),
                 (2000, 2000, 20), (-2000, 2000, 20))
    angle = -60.0
    alt = 20
    circuit_radius = 100
    # [+ forward - back, + right -left, alt]
    # straight_line = [[-2000, 0, 100], [1000, 0, 100], [1000, -2000, 100], [2, -1, 100]]
    tight_circuit = [[0, 0, 0], [4000, 0, alt], [4000, circuit_radius, alt], [0, circuit_radius, alt], [0, 0, alt],
                     [4000, 0, alt], [4000, circuit_radius, alt], [0, circuit_radius, alt]]

    origin = [env.sim[prp.initial_latitude_geod_deg] * 111120.0, env.sim[prp.initial_longitude_geoc_deg] * 111120.0]
    path = env.transform_path(tight_circuit, angle, origin)
    print(path)
    env.simulation_loop(path)
    env.generate_figures()
    print("Simulation Ended")


if __name__ == '__main__':
    simulate()
