from jsbsim_conn import Simulator, Autopilot
from image_processing import get_images
from debug_utils import DebugGraphs


def main():
    sim = Simulator()
    # ap = Autopilot()
    # controls = [0, 0, 1, 0]  # controls with [eatr] type definition
    # sim.stream_controls(controls)
    graph = DebugGraphs(sim)
    for _ in range(500):
        sim.set_airsim()
        # ap.trim_off()
        # ap.heading_hold(0)
        sim.advance_stream_n(10)
        graph.get_time_data()
        graph.get_pos_data()
        graph.get_angle_data()
    print('Finished for loop!!!')
    graph.basic_plot()
    return


main()
