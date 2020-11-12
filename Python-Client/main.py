import airsim
from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from debug_utils import *
import jsbsim_properties as prp
from simple_pid import PID
from autopilot import X8Autopilot
from Navigation import WindEstimation
from image_processing import AirSimImages, SemanticImageSegmentation


def main():
    sim = Simulation(120, x8, None, 0)
    ap = X8Autopilot(sim)
    graph = DebugGraphs(sim)
    debug_aero = DebugFDM(sim)
    wind_estimate = WindEstimation(sim)
    # images = AirSimImages(sim)
    image_segmentation = SemanticImageSegmentation(sim)
    flag1 = False
    over = False
    for _ in range(20000):
        sim.update_airsim()
        collision_info = sim.get_collision_info()
        print(collision_info.has_collided)
        # image_segmentation.get_png_image(airsim.ImageType.Scene)
        image_segmentation.get_segmented_image_plot()
        # ap.pitch_hold(0.05)
        # ap.heading_hold(0.0)
        ap.airspeed_hold_w_throttle(50.0)
        # ap.altitude_hold(100)
        print(sim.get_local_position())
        # circuit_profile = ((0, 0, 1000), (4000, 0, 1000), (4000, 4000, 1000), (0, 4000, 1000), (0, 0, 20),
        #                    (4000, 0, 20), (4000, 4000, 20))
        ice_profile = ((0, 0, 0), (1200, 0, 0), (1300, 150, -100), (0, 0, -200), (100, 100, -200))
        if not over:
            # wind_estimate.wind_average(n=5)
            over = ap.arc_path(ice_profile, 200)
        if over:
            print('over and out!')
            break
        graph.get_abs_pos_data()
        graph.get_airspeed()
        graph.get_alpha()
        graph.get_control_data()
        graph.get_time_data()
        graph.get_pos_data()
        graph.get_angle_data()
        sim.run()
    print('Simulation ended')
    graph.control_plot()
    graph.trace_plot_abs()
    return


main()
