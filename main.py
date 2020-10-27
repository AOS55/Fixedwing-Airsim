from jsbsim_conn import Simulator
from image_processing import get_images
from debug_utils import DebugGraphs


def main():
    sim = Simulator()
    # controls = [0, 0, 1, 0]  # controls with [eatr] type definition
    # sim.stream_controls(controls)
    # graph = DebugGraphs(sim)
    for _ in range(100):
        sim.set_airsim()
        sim.advance_stream_n(1)
        # graph.get_time_data()
        # graph.get_pos_data()
    print('Finished for loop!!!')
    # graph.basic_plot()
    return


main()
