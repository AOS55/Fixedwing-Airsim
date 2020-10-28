import airsim

"""Should be deleted, just to test unknowns"""

# pitch = [0] * 10
# roll = [0] * 10
# yaw = [0, 0, 0, 0, 0, 6.1, 6.1, 6.0, 6.0, 6.0]
#
# alt = [10] * 10
# long = [0] * 10
# lat = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 25.0, 25.5]
#
# client = airsim.VehicleClient()
# client.confirmConnection()
#
# for t in range(10):
#     pose = client.simGetVehiclePose()
#     pose.position.x_val = lat[t]
#     pose.position.y_val = long[t]
#     pose.position.z_val = - alt[t]
#     pose.orientation = airsim.to_quaternion(pitch[t], roll[t], yaw[t])
#     client.simSetVehiclePose(pose, True)


