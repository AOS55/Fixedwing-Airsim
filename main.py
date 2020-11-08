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
    flag1 = False
    flag2 = False
    flag3 = False
    flag4 = False
    over = False
    for _ in range(90000):
        # sim.update_airsim()
        ap.pitch_hold(0.05)
        # if sim[prp.sim_time_s] > 20:
        #     sim[prp.aileron_cmd] = 0.05
        if sim[prp.sim_time_s] > 20.5:
            # ap.roll_hold(0.0)
            # sim[prp.throttle_cmd] = 0.0
            # ap.vs_hold_w_throttle(500)
            ap.airspeed_hold_w_throttle(50.0)
            # if not over:
            circuit_profile = ((0, 0), (4000, 0), (4000, 4000), (0, 4000), (0, 0), (4000, 0), (4000, 4000))
            #     over = ap.track_to_profile(circuit_profile)
            point = (500, 500)
            if not flag1:
                flag1 = ap.fillet_path(circuit_profile, 500)

            # if not flag1:
            #     flag1 = ap.track_to_target(1000, 1, 1000)
            #     # print(flag1)
            # if not flag2:
            #     flag2 = ap.track_to_target(1, 500, 1000)
            #     # print(flag2)
            # if not flag3:
            #     flag3 = ap.track_to_target(-1000, 1, 1000)
            # if not flag4:
            #     flag4 = ap.track_to_target(1, -500, 1000)
            # if flag1:
            #     print('Flag1')
            # if flag2:
            #     print('Flag2')
            # if flag3:
            #     print('Flag3')
            # if flag4:
            #     print('Flag4')
            # ap.altitude_hold(100.0)
            # ap.heading_hold(190.0)
        if 19.5 < sim[prp.sim_time_s] < 1200:
            graph.get_abs_pos_data()
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
    graph.trace_plot_abs()
    return


main()
