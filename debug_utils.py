import matplotlib.pyplot as plt
import numpy as np
import jsbsim_properties as prp
from jsbsim_conn import Simulator


class DebugGraphs():
    def __init__(self, sim):
        self.sim = sim
        self.time = []
        self.lat = []
        self.long = []
        self.alt = []
        self.yaw = []
        self.pitch = []
        self.roll = []

    def get_time_data(self):
        self.time.append(self.sim.tn.get_property_stream(prp.sim_time_s))

    def get_pos_data(self):
        self.lat.append(self.sim.tn.get_property_stream(prp.lng_travel_m))
        self.long.append(self.sim.tn.get_property_stream(prp.lat_travel_m))
        self.alt.append(self.sim.tn.get_property_stream(prp.altitude_sl_ft))

    def get_angle_data(self):
        self.yaw.append(self.sim.tn.get_property_stream(prp.heading_deg))
        self.pitch.append(self.sim.tn.get_property_stream(prp.pitch_rad))
        self.roll.append(self.sim.tn.get_property_stream(prp.roll_rad))

    def basic_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time, self.roll)
        ax.plot(self.time, self.pitch)
        ax.plot(self.time, self.yaw)
        plt.show()

