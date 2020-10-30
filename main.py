from jsbsim_simulator import Simulation
from jsbsim_aircraft import Aircraft, cessna172P, ball, x8
from debug_utils import *
import jsbsim_properties as prp
from simple_pid import PID
from autopilot import X8Autopilot


def main():
    sim = Simulation(120, x8, None, 0)
    ap = X8Autopilot(sim)
    graph = DebugGraphs(sim)
    debug_aero = DebugFDM(sim)
    for _ in range(10000):
        # sim.update_airsim()
        # ap.pitch_hold(0.05)
        if sim[prp.sim_time_s] > 20:
            sim[prp.throttle_cmd] = 0.02
        if sim[prp.sim_time_s] > 20.5:
            sim[prp.throttle_cmd] = 0.0
            ap.airspeed_hold_w_throttle(60.0)
            ap.altitude_hold(1200.0)
        if 19 < sim[prp.sim_time_s] < 40:
            graph.get_airspeed()
            graph.get_alpha()
            graph.get_control_data()
            graph.get_time_data()
            graph.get_pos_data()
            graph.get_angle_data()
        sim.run()
    print('Simulation ended')
    # graph.pos_plot()
    # graph.att_plot()
    # graph.lift_plot()
    # graph.pitch_plot()
    graph.control_plot()
    return


main()
