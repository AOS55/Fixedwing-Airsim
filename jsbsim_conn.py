from jsbsim_server import run_server
import jsbsim_properties as prp
import airsim
import math


class Simulator:
    def __init__(self, tn=run_server()):
        self.tn = tn
        self.tn.print_command('info')   # kickstart the connection, don't know why this is needed
        self.client = self.airsim_connect()

    def stream_position(self):
        lat = self.tn.get_property_stream(prp.lng_travel_m)
        long = self.tn.get_property_stream(prp.lat_travel_m)
        alt = self.tn.get_property_stream(prp.altitude_sl_ft)
        position = [lat, long, alt]
        return position

    def stream_orientation(self):
        pitch = self.tn.get_property_stream(prp.pitch_rad)
        roll = self.tn.get_property_stream(prp.roll_rad)
        yaw = self.tn.get_property_stream(prp.heading_deg) * (math.pi / 180)
        orientation = [pitch, roll, yaw]
        return orientation

    def get_time(self):
        time = self.tn.get_property_value('simulation/sim-time-sec')
        return time

    @staticmethod
    def airsim_connect():
        client = airsim.VehicleClient()
        client.confirmConnection()
        return client

    def set_airsim(self):
        pose = self.client.simGetVehiclePose()
        position = self.stream_position()
        pose.position.x_val = position[0]
        pose.position.y_val = position[1]
        pose.position.z_val = - position[2]
        euler_angles = self.stream_orientation()
        pose.orientation = airsim.to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
        self.client.simSetVehiclePose(pose, True)
        return

    def stream_controls(self, control_cmd):
        self.tn.set_property_stream(prp.elevator_cmd, control_cmd[0])
        self.tn.set_property_stream(prp.aileron_cmd, control_cmd[1])
        self.tn.set_property_stream(prp.throttle_cmd, control_cmd[2])
        self.tn.set_property_stream(prp.rudder_cmd, control_cmd[3])
        return

    def advance_stream_n(self, n):
        self.tn.iterate_n(n)

    def advance_n(self, n):
        self.tn.send_command('iterate '+str(n))
        return

    def advance(self):
        self.tn.send_command('iterate 1')
        return


class Autopilot(Simulator):
    def __init__(self):
        super().__init__()

    def heading_hold(self, hdg):
        self.tn.set_property_stream(prp.heading_switch, 1)
        self.tn.set_property_stream(prp.heading_des, hdg)

    def heading_hold_off(self):
        self.tn.set_property_stream(prp.heading_switch, 0)

    def level_hold(self, level):
        self.tn.set_property_stream(prp.level_switch, 1)
        self.tn.set_property_stream(prp.level_des, level)

    def level_hold_off(self):
        self.tn.set_property_stream(prp.level_switch, 0)

    def trim_on(self):
        self.tn.set_property_stream(prp.trim_switch, 1)

    def trim_off(self):
        self.tn.set_property_stream(prp.trim_switch, 0)




