import airsim
import math
import matplotlib.pyplot as plt

"""Should be deleted, just to test unknowns, should probably implement actual tests !!!"""

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

airspeed = 25.653529915395765
heading = 20 * (math.pi / 180)
old_cur = (1881.965612819288, 265.3750003228467)
cur = (1882.1719505281862, 265.36504842964456)
dt = 0.0083333333

track_vector = (old_cur[0] - cur[0], old_cur[1] - cur[1])
track_angle = math.atan2(track_vector[1], track_vector[0]) - math.pi
if track_angle < 0:
    track_angle = track_angle + (2 * math.pi)
ground_speed = math.sqrt(pow(track_vector[0], 2) + pow(track_vector[1], 2)) / dt

print(track_angle * (180 / math.pi), ground_speed)

wind_speed = math.sqrt(pow(airspeed, 2) + pow(ground_speed, 2) -
                       (2 * airspeed * ground_speed * math.cos(track_angle - heading)))
try:
    wind_angle = math.pi + track_angle + \
                 math.asin((airspeed * math.sin(track_angle - heading)) / wind_speed)
    if wind_angle > 2 * math.pi:
        wind_angle = wind_angle - (2 * math.pi)
    if wind_angle < 0:
        wind_angle = wind_angle + (2 * math.pi)
except ZeroDivisionError:
    wind_angle = 0.0
print(airspeed, wind_speed, track_angle * (180 / math.pi), heading * (180 / math.pi), math.asin((airspeed * math.sin(track_angle - heading)) / wind_speed) * (180 / math.pi))

print(wind_speed, wind_angle * (180 / math.pi))

fig, ax = plt.subplots()
