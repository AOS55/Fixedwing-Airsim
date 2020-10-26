from jsbsim_server import run_server
import airsim


class Simulator:
    def __init__(self, tn=run_server()):
        self.tn = tn
        self.tn.print_command('info')   # kickstart the connection, don't know why this is needed

    def get_position(self):
        lat = self.tn.get_property_value('position/distance-from-start-lat-mt')
        long = self.tn.get_property_value('position/distance-from-start-lon-mt')
        alt = self.tn.get_property_value('position/distance-from-start-mag-mt')
        position = [lat, long, alt]
        return position

    def get_orientation(self):
        yaw = self.tn.get_property_value('attitude/heading-true-rad')
        pitch = self.tn.get_property_value('attitude/pitch-rad')
        roll = self.tn.get_property_value('attitude/roll-rad')
        orientation = [pitch, roll, yaw]
        return orientation

    def get_time(self):
        time = self.tn.get_property_value('simulation/sim-time-sec')
        return time

    # def set_controls(self, control):
    #     self.tn.set_property_value('fcs/aileron-cmd-norm,')

    def advance_n(self, n):
        self.tn.send_command('iterate '+str(n))

    def advance(self):
        self.tn.send_command('iterate 1')
        return

def main():
    sim = Simulator()
    client = airsim.VehicleClient()
    client.confirmConnection()
    pose = client.simGetVehiclePose()
    euler_angles = airsim.to_eularian_angles(client.simGetVehiclePose().orientation)
    for _ in range(10):
        pose.position.x_val = sim.get_position()[0]
        pose.position.y_val = sim.get_position()[1]
        pose.position.z_val = - sim.get_position()[2]
        euler_angles = sim.get_orientation()
        pose.orientation = airsim.to_quaternion(euler_angles[0], euler_angles[1], euler_angles[2])
        sim.tn.print_command('get fcs/throttle-cmd-norm')
        print("x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
        client.simSetVehiclePose(pose, True)
        sim.advance_n(5)


main()
