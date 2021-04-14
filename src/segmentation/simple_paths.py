import airsim
from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from debug_utils import *
import navigation as navigation
import jsbsim_properties as prp
from autopilot import X8Autopilot
from image_processing import AirSimImages
from typing import Dict
from report_diagrams import ReportGraphs
from datetime import datetime
import os
import numpy as np
import cv2
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

    def simulation_loop(self, profile: tuple, angle: float, alt: float, circuit_radius: float, circuit_type: str) -> \
            None:
        """
        Runs the closed loop simulation and updates to airsim simulation based on the class level definitions

        :param profile: a tuple of tuples of the aircraft's profile in (lat [m], long [m], alt [feet])
        :param angle: angle relative to origin [degrees]
        :param alt: altitude above sea level [feet]
        :param circuit_radius: radius of circuit flown [m]
        :param circuit_type: the name of the type of circuit flown
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
                self.make_metadata_json(self.dataset_name, image_id, angle, alt, circuit_radius, circuit_type)
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
                    # TODO: seg method just skips forward one until we get there would be better go to end of images
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
                    # TODO: image method just skips forward one until we get there would be better go to end of images
                cv2.imwrite(os.path.join(image_path, str(image_id) + ".png"), img_rgb)
        return seg_id, image_id

    def store_flight_data(self, dataset: str, flight_id: int) -> None:
        """
        Store flight data from the simulation as a JSON object

        :param dataset: the name of the directory where the dataset is to be stored
        :param flight_id: the index of the JSON object as related to the image
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

    def make_metadata_json(self, dataset: str, metadata_id: int, angle: float, alt: float, circuit_radius: float,
                           circuit_type: str) -> None:
        """
        Make and store a document containing the metadata for each image pair stored in the file system

        :param dataset: the name of the directory where the dataset is to be stored
        :param metadata_id: the index of the JSON object as related to the image
        :param angle: angle relative to origin [degrees]
        :param alt: altitude above sea level [feet]
        :param circuit_radius: radius of circuit flown [m]
        :param circuit_type: the name of the type of circuit flown
        :return: None
        """
        dirname = os.path.dirname(__file__)  # get the location of the root directory
        dirname = os.path.join(dirname, '../..')  # move out of segmentation source directory
        dirname = os.path.join(dirname, 'data/segmentation-datasets')  # go into segmentation-dataset dir
        dirname = os.path.join(dirname, dataset)  # go into segmentation specific dataset
        metadata_path = os.path.join(dirname, 'metadata')  # make a path to flight_data dir specifically

        rel_dataset = os.path.join('/data/segmentation-datasets', dataset)
        cur_seg = os.path.join(rel_dataset + "/segmentation_masks/" + str(metadata_id) + ".png")  # rel path from root
        # to seg mask
        cur_image = os.path.join(rel_dataset + "/images/" + str(metadata_id) + ".png")  # rel path from root to image
        cur_fd = os.path.join(rel_dataset + "/flight_data/" + str(metadata_id) + ".json")  # rel path from root to
        # flight data

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        metadata_dict = {
            'image_id': metadata_id,
            'sim_time': self.sim.get_time(),
            'mask_loc': cur_seg,
            'image_loc': cur_image,
            'fd_loc': cur_fd,
            'utc_time': current_time,
            'plan_angle': angle,
            'plan_altitude': alt,
            'circuit_radius': circuit_radius,
            'circuit_type': circuit_type
        }
        json_file_name = os.path.join(metadata_path, str(metadata_id) + ".json")
        with open(json_file_name, 'w') as json_file:
            json.dump(metadata_dict, json_file)


class Simulate:

    def __init__(self,
                 dataset_name: str = "default-dataset",
                 sim_time: float = 750,
                 display_graphics: bool = True,
                 airsim_frequency_hz: float = 0.2
                 ):
        self.dataset_name = dataset_name
        self.sim_time = sim_time
        self.display_graphics = display_graphics
        self.airsim_frequency_hz = airsim_frequency_hz
        # definition of the initial runway conditions
        self.runway_start = {
            prp.initial_altitude_ft: 100,
            prp.initial_latitude_geod_deg: -1000.0000000000002 / 111120.0,
            prp.initial_longitude_geoc_deg: 1732.0508075688772 / 111120.0,
            prp.initial_u_fps: 50.0,
            prp.initial_w_fps: 0.0,
            prp.initial_heading_deg: 300.0,
            prp.initial_roc_fpm: 0.0,
            prp.all_engine_running: -1
        }
        self.env = self.setup_simulator_environment()

    def setup_simulator_environment(self) -> ImagePath:
        """
        Setup an instance of the ImagePath class to collect images

        :param dataset_name: the name of the directory all the images & JSON information will be stored in
        :param sim_time: the time the simulation will run for until it is terminated
        :param display_graphics: whether or not to display graphics, if set False no images will be collected
        :param airsim_frequency_hz: how often images will be collected 1/f = image rate e.g. 1/0.2 = 5s per image
        :return: instance of image path class
        """
        # setup an environment
        env = ImagePath(sim_time=self.sim_time,
                        display_graphics=self.display_graphics,
                        init_conditions=self.runway_start,
                        airsim_frequency_hz=self.airsim_frequency_hz,
                        dataset_name=self.dataset_name)
        return env

    def simulate_circuit(self, circuit_radius: float = 100, alt: float = 20, angle: float = -60.0) -> None:
        """
        Fly a circuit to collect images around a location defined in the simulator_env
        :param env: an instance of the ImagePath class
        :param circuit_radius: the distance between the final/upwind and downwind portion of the circuit
        :param alt: the altitude to fly the circuit at
        :param angle: the angle to fly the circuit at, -60.0 lines you up with the Lydd runway
        :return: None
        """
        # [+ forward - back, + right -left, alt]
        tight_circuit = [[0, 0, 0], [4000, 0, alt], [4000, circuit_radius, alt], [0, circuit_radius, alt], [0, 0, alt],
                         [4000, 0, alt], [4000, circuit_radius, alt], [0, circuit_radius, alt]]

        origin = [self.env.sim[prp.initial_latitude_geod_deg] * 111120.0, self.env.sim[prp.initial_longitude_geoc_deg] *
                  111120.0]
        path = self.env.transform_path(tight_circuit, angle, origin)
        print(path)
        self.env.simulation_loop(path, angle=angle, alt=alt, circuit_radius=circuit_radius, circuit_type="tight_circuit")
        self.env.generate_figures()
        print("Circuit finished Ended")


def fly_several_circuits(dataset_name: str, min_alt: float = 20, max_alt: float = 500, incr_alt: float = 50,
                         min_width: float = 100, max_width: float = 1000, incr_width: float = 100) -> None:
    """
    Simulates flying multiple circuits under different conditions

    :param dataset_name: name of dataset to save names into
    :param min_alt: minimum altitude to be flown [feet]
    :param max_alt: maximum altitude to be flown [feet]
    :param incr_alt: increment of altitude to be flown [feet]
    :param min_width: minimum width of circuit to be flown [m]
    :param max_width: maximum width of circuit to be flown [m]
    :param incr_width: increment of circuit width to be flown [m]
    :return: None
    """
    alts = [x for x in range(int(min_alt), int(max_alt), int(incr_alt))]
    widths = [x for x in range(int(min_width), int(max_width), int(incr_width))]
    for alt in alts:
        for width in widths:
            sim = Simulate(dataset_name)
            sim.simulate_circuit(width, alt)
    print("Simulation finished")


if __name__ == '__main__':
    fly_several_circuits(dataset_name="large-multicct")
