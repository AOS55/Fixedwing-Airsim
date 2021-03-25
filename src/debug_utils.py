import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import src.jsbsim_properties as prp
import math


class DebugGraphs:
    def __init__(self, sim):
        self.sim = sim
        self.time = []
        self.lat = []
        self.long = []
        self.lat_abs = []
        self.long_abs = []
        self.alt = []
        self.yaw = []
        self.pitch = []
        self.roll = []
        self.airspeed = []
        self.vs = []

        self.alpha = []

        self.clo = []
        self.clalpha = []
        self.clq = []
        self.clde = []

        self.cmo = []
        self.cmalpha = []
        self.cmq = []
        self.cmde = []

        self.aileron_cmd = []
        self.elevator_cmd = []
        self.throttle_cmd = []
        self.rudder_cmd = []

        self.aileron_left = []
        self.aileron_right = []
        self.aileron_combined = []
        self.elevator = []
        self.throttle = []
        self.rudder = []

        self.p = []
        self.q = []
        self.r = []

    def get_time_data(self):
        self.time.append(self.sim.get_time())

    def get_pos_data(self):
        self.lat.append(self.sim.get_local_position()[0])
        self.long.append(self.sim.get_local_position()[1])
        self.alt.append(self.sim.get_local_position()[2])

    def get_abs_pos_data(self):
        self.lat_abs.append(self.sim[prp.lat_geod_deg])
        self.long_abs.append(self.sim[prp.lng_geoc_deg])

    def get_angle_data(self):
        self.pitch.append(self.sim.get_local_orientation()[0])
        self.roll.append(self.sim.get_local_orientation()[1])
        self.yaw.append(self.sim.get_local_orientation()[2] * (180 / math.pi))

    def get_lift_data(self):
        # normalized to ignore the aircrafts velocity
        self.clo.append(self.sim[prp.Clo] / self.sim[prp.qbar_area])
        self.clalpha.append(self.sim[prp.Clalpha] / self.sim[prp.qbar_area])
        self.clq.append(self.sim[prp.Clq] / self.sim[prp.qbar_area])
        self.clde.append(self.sim[prp.ClDe] / self.sim[prp.qbar_area])

    def get_pitch_data(self):
        self.cmo.append(self.sim[prp.Cmo] / self.sim[prp.qbar_area])
        self.cmalpha.append(self.sim[prp.Cmalpha] / self.sim[prp.qbar_area])
        self.cmq.append(self.sim[prp.Cmq] / self.sim[prp.qbar_area])
        self.cmde.append(self.sim[prp.CmDe] / self.sim[prp.qbar_area])

    def get_control_data(self):
        self.elevator_cmd.append(self.sim[prp.elevator_cmd])
        self.aileron_cmd.append(self.sim[prp.aileron_cmd])
        self.throttle_cmd.append(self.sim[prp.throttle_cmd])
        self.rudder_cmd.append(self.sim[prp.rudder_cmd])
        self.elevator.append(self.sim[prp.elevator_rad])
        self.aileron_left.append(self.sim[prp.aileron_left_rad])
        self.aileron_right.append(self.sim[prp.aileron_right_rad])
        self.aileron_combined.append(self.sim[prp.aileron_combined_rad])
        self.throttle.append(self.sim[prp.throttle])
        self.rudder.append(self.sim[prp.rudder_rad])

    def get_rate_data(self):
        self.p.append(self.sim[prp.p_radps])
        self.q.append(self.sim[prp.q_radps])
        self.r.append(self.sim[prp.r_radps])

    def get_alpha(self):
        self.alpha.append(self.sim[prp.alpha])

    def get_airspeed(self):
        self.airspeed.append(self.sim[prp.airspeed] * 0.5925)
        self.vs.append(self.sim[prp.v_down_fps] * -1 * 60)  # multiplied to fpm from fps

    def pos_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.lat)
        ax.plot(self.time, self.long)
        ax.plot(self.time, self.alt)
        plt.show()

    def att_plot(self):
        fig, ax = plt.subplots()
        # ax.plot(self.time, self.pitch)
        ax.plot(self.time, self.roll)
        ax.plot(self.time, self.yaw)
        plt.show()

    def lift_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.alpha, self.clo, color='green', marker='.')
        ax.plot(self.alpha, self.clalpha, color='blue', marker='.')
        ax.plot(self.alpha, self.clq, color='orange', marker='.')
        ax.plot(self.alpha, self.clde, color='red', marker='.',)
        plt.show()

    def pitch_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.alpha, self.cmo, color='green', marker='.')
        ax.plot(self.alpha, self.cmalpha, color='blue', marker='.')
        ax.plot(self.alpha, self.cmq, color='orange', marker='.')
        ax.plot(self.alpha, self.cmde, color='red', marker='.',)
        plt.show()

    def roll_rate_plot(self):
        fig, ax = plt.subplots()
        ax.set_title('roll rate')
        ax.plot(self.time, self.p)
        plt.show()

    def pitch_rate_plot(self):
        fig, ax = plt.subplots()
        ax.set_title('pitch rate')
        ax.plot(self.time, self.q)
        plt.show()

    def control_plot(self):
        fig, ax = plt.subplots()
        ax.set_title('Throttle Control Plot')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Control deflection [-]')
        # ax.plot(self.time, self.elevator_cmd)
        # ax.plot(self.time, self.aileron_cmd)
        # ax.plot(self.time, self.throttle_cmd)
        # ax.plot(self.time, self.rudder_cmd)
        # ax.plot(self.time, self.aileron_left)
        # ax.plot(self.time, self.aileron_right)
        # ax.plot(self.time, self.aileron_combined)
        # ax.plot(self.time, self.roll)
        # ax.plot(self.time, [x * 10 for x in self.elevator])
        # ax.plot(self.time, [x * (180.0 / math.pi) for x in self.pitch])
        # ax.plot(self.time, self.throttle)
        # ax.plot(self.time, self.rudder)
        # ax.plot(self.time, self.airspeed)
        ax.plot(self.time, self.alt)
        # ax.plot(self.time, self.vs)
        # ax.plot(self.time, self.lat)
        # ax.plot(self.time, self.long)
        # ax.plot(self.time, self.yaw)
        plt.savefig("Control_plot.eps", format='eps')
        plt.show()

    def trace_plot(self):
        fig, ax = plt.subplots()
        ax.set_title('Trace Plot')
        ax.set_xlabel('Latitude [degs]')
        ax.set_ylabel('Longitude [degs]')
        ax.plot(self.lat, self.long)
        plt.show()

    def trace_plot_abs(self):
        fig, ax = plt.subplots()
        ax.set_title('Trace Plot')
        ax.set_xlabel('Latitude [degs]')
        ax.set_ylabel('Longitude [degs]')
        long_m = [x * 111120.0 for x in self.long_abs]
        lat_m = [x * 111120.0 for x in self.lat_abs]
        ax.plot(long_m, lat_m)
        plt.savefig("Trace_plot")
        plt.show()

    def three_d_scene(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title('3D plot')
        ax.set_xlabel('Latitude [degs]')
        ax.set_ylabel('Longitude [degs]')
        ax.set_zlabel('Altitude [m]')
        zline = [x / 3.28 for x in self.alt]
        xline = self.lat_abs
        yline = self.long_abs
        ax.plot3D(xline, yline, zline, 'gray')
        plt.savefig("threed")
        plt.show()


class DebugFDM:
    def __init__(self, sim):
        self.sim = sim

    def get_lift_values(self):
        print('vt: ', self.sim[prp.airspeed] * 0.5925)
        # print('Sw: ', self.sim[prp.Sw])
        # print('density: ', self.sim[prp.rho])
        # print('qbar_area: ', self.sim[prp.qbar_area])
        # print('ci2vel: ', self.sim[prp.ci2vel])
        print('alpha: ', self.sim[prp.alpha])
        print('Clo: ', self.sim[prp.Clo])
        print('Clalpha: ', self.sim[prp.Clalpha])
        print('Clq: ', self.sim[prp.Clq])
        print('ClDe: ', self.sim[prp.ClDe])
        total_lift = self.sim[prp.Clo] + self.sim[prp.Clalpha] + self.sim[prp.Clq] + self.sim[prp.ClDe]
        print('Lift = ', total_lift)

    def get_roll_values(self):
        print('vt: ', self.sim[prp.airspeed] * 0.5925)
        print('p: ', self.sim[prp.p_radps])

    def get_pitch_values(self):
        print('Cmo: ', self.sim[prp.Cmo])
        print('Cmalpha: ', self.sim[prp.Cmalpha])
        print('Cmq: ', self.sim[prp.Cmq])
        print('CmDe: ', self.sim[prp.CmDe])
        print('q: ', self.sim[prp.q_radps])
        print('ROC: ', self.sim[prp.v_down_fps] * -1 * 60)
