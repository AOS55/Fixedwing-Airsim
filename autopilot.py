from simple_pid import PID
import jsbsim_properties as prp
from jsbsim_simulator import Simulation
from scipy import interpolate
import math
from Navigation import Navigation, LocalNavigation


# Should this be derived from simulation ?
# def __init__(self):
#     super().__init__()
class C172Autopilot:
    def __init__(self, sim):
        self.sim = sim

    def wing_leveler(self):
        error = self.sim[prp.roll_rad]
        kp = 50.0
        ki = 5.0
        kd = 17.0
        pid = PID(kp, ki, kd)
        output = pid(error)
        self.sim[prp.aileron_cmd] = output

    def hdg_hold(self, hdg):
        error = hdg - self.sim[prp.heading_deg]
        # Limit error to within 180 degrees (left or right)
        if error > 180:
            error = error - 180
        if error < 180:
            error = error + 180
        # Saturate error signal gain to be a maximum of 30 degrees
        if error < -30:
            error = -30
        if error > 30:
            error = 30
        # Convert error signal from degrees to radians
        error = error * (math.pi / 180)
        # Implementing a lag compensator as a single integrator (don't know how to do lag)
        c = 0.5
        hdg_lag = PID(0, c, 0)
        roll_error = hdg_lag(error) - self.sim[prp.roll_rad]
        kp = 6.0
        ki = 0.13
        kd = 6.0
        roll_pid = PID(kp, ki, kd)
        output = roll_pid(roll_error)
        self.sim[prp.aileron_cmd] = output

    def level_hold(self, level):
        error = level - self.sim[prp.altitude_sl_ft]
        # print('level hold error: ', error)
        # Limit climb error to a maximum of 100'
        if error > 100:
            error = 100
        if error < -100:
            error = -100
        # Convert error to percentage of maximum
        error = error/100
        # print('percentage error: ', error)
        # Lag desired climb rate (for stability) as a single integrator
        # c = 1.0
        # vs_lag = PID(0, c, 0)
        # error = vs_lag(error)
        # Gain scheduled climb rate
        ref_alt = [0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
                   7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0]
        vs_gain = [0.12, 0.11, 0.10, 0.096, 0.093, 0.086, 0.078,
                   0.069, 0.061, 0.053, 0.045, 0.037, 0.028]
        climb_gain_scheduling = interpolate.interp1d(ref_alt, vs_gain)
        vs_dem = error * climb_gain_scheduling(self.sim[prp.altitude_sl_ft])
        vs_error = vs_dem - self.sim[prp.altitude_rate_fps]
        # print('vertical speed error: ', vs_error)
        # Rate PID controller
        kp = 0.01
        ki = 0.00015
        kd = 0.0003
        vs_pid = PID(kp, ki, kd)
        output = vs_pid(vs_error)
        self.sim[prp.elevator_cmd] = output
        # print('elevator command: ', output)


class X8Autopilot:
    def __init__(self, sim):
        self.sim = sim
        self.nav = None
        self.orbit_nav = None
        self.track_bearing = 0
        self.track_bearing_in = 0
        self.track_bearing_out = 0
        self.track_distance = 0
        self.flag = False
        self.track_id = -1
        self.state = 0

    def pitch_hold(self, pitch_comm):
        # Nichols Ziegler tuning Pcr = 0.25s, Kcr = 7.5, PI chosen
        error = pitch_comm - self.sim[prp.pitch_rad]
        kp = 3.4
        ki = 0.208
        kd = 0.0
        controller = PID(kp, ki, kd)
        output = controller(error)
        self.sim[prp.elevator_cmd] = output

    def roll_hold(self, roll_comm):
        # Nichols Ziegler tuning Pcr = 0.29, Kcr = 0.0380, PID chosen
        error = roll_comm - self.sim[prp.roll_rad]
        kp = 0.0228
        ki = 0.145
        kd = 0.03625
        controller = PID(kp, ki, kd)
        output = - controller(error)
        self.sim[prp.aileron_cmd] = output

    def heading_hold(self, heading_comm):
        # Attempted Nichols-Ziegler with Pcr = 0.048, Kcr=1.74, lead to a lot of overshoot
        error = heading_comm - self.sim[prp.heading_deg]
        # Ensure the aircraft always turns the shortest way round
        if error < -180:
            error = error + 360
        if error > 180:
            error = error - 360
        kp = 0.0085
        ki = 0.0
        kd = 0.05
        heading_controller = PID(kp, ki, kd)
        output = heading_controller(-error)
        # Prevent over-bank +/- 30 degrees
        if output < - 30 * (math.pi / 180):
            output = - 30 * (math.pi / 180)
        if output > 30 * (math.pi / 180):
            output = 30 * (math.pi / 180)
        self.roll_hold(output)

    def airspeed_hold_w_throttle(self, airspeed_comm):
        # Appears fine with simple proportional controller, light airspeed instability at high speed (100kts)
        error = airspeed_comm - (self.sim[prp.airspeed] * 0.5925)
        kp = 0.022
        ki = 0.0
        kd = 0.0
        airspeed_controller = PID(kp, kd, ki)
        output = airspeed_controller(-error)
        self.sim[prp.throttle_cmd] = output

    def altitude_hold(self, altitude_comm):
        # Tuned from level off works up to around 100kts used Nichols-Ziegler with Pcr = 0.1, Kcr=0.19
        error = altitude_comm - self.sim[prp.altitude_sl_ft]
        kp = 0.11
        ki = 0.05
        kd = 0.03
        altitude_controller = PID(kp, ki, kd)
        output = altitude_controller(-error)
        # prevent excessive pitch +/- 15 degrees
        if output < - 15 * (math.pi / 180):
            output = - 15 * (math.pi / 180)
        if output > 15 * (math.pi / 180):
            output = 15 * (math.pi / 180)
        self.pitch_hold(output)

    def home_to_target(self, target_northing, target_easting, target_alt):
        if self.nav is None:
            # initialize target
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(target_northing, target_easting)
            self.flag = False
        if self.nav is not None:
            if not self.flag:
                # fly to target
                bearing = self.nav.bearing() * 180.0 / math.pi
                if bearing < 0:
                    bearing = bearing + 360
                distance = self.nav.distance()
                if distance < 100:
                    self.flag = True
                    self.nav = None
                    return self.flag
                # heading_error = bearing - self.sim[prp.heading_deg]
                # heading_error = 1.0 * heading_error
                # self.heading_hold(self.sim[prp.heading_deg] + heading_error)
                self.heading_hold(bearing)
                self.altitude_hold(target_alt)
                # print('Demanded heading: ', bearing, 'Actual heading: ', self.sim[prp.heading_deg])
                # print('Distance to target:', distance)

    def track_to_target(self, target_northing, target_easting, target_alt):
        if self.nav is None:
            # initialize target and track
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(target_northing, target_easting)
            self.track_bearing = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing < 0:
                self.track_bearing = self.track_bearing + 360.0
            self.track_distance = self.nav.distance()
            self.flag = False
        if self.nav is not None:
            # position relative to target
            bearing = self.nav.bearing() * 180.0 / math.pi
            if bearing < 0:
                bearing = bearing + 360
            distance = self.nav.distance()
            off_tk_angle = self.track_bearing - bearing
            x_track = self.nav.x_track_error(distance, off_tk_angle)
            distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
            # use a controller to regulate the closure rate relative to the track
            error = off_tk_angle * distance_to_go
            kp = 0.01
            ki = 0.0
            kd = 0.0
            closure_controller = PID(kp, ki, kd)
            # print(closure_controller(-error), bearing)
            heading = closure_controller(-error) + bearing
            if distance < 200:
                self.flag = True
                self.nav = None
                return self.flag
            self.heading_hold(heading)
            self.altitude_hold(target_alt)
            # print('Demanded heading: ', heading, 'Actual heading: ', self.sim[prp.heading_deg])
            # print('Distance to target:', distance)

    def track_to_profile(self, profile):
        if self.nav is None:
            self.track_id = self.track_id + 1
            if self.track_id == len(profile) - 1:
                print('hit flag')
                self.flag = True
                return self.flag
            point_a = profile[self.track_id]
            point_b = profile[self.track_id + 1]
            # initialize target and track
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(point_b[0] - point_a[0], point_b[1] - point_a[1])
            print(point_b[0] - point_a[0], point_b[1] - point_a[1])
            self.track_bearing = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing < 0:
                self.track_bearing = self.track_bearing + 360.0
            self.track_distance = self.nav.distance()
            self.flag = False
        if self.nav is not None:
            bearing = self.nav.bearing() * 180.0 / math.pi
            if bearing < 0:
                bearing = bearing + 360
            distance = self.nav.distance()
            off_tk_angle = bearing - self.track_bearing
            if off_tk_angle > 180:
                off_tk_angle = off_tk_angle - 360.0
            # scale response with distance from target
            distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
            if distance_to_go > 3000:
                distance_to_go = 3000
            heading = (8 * 0.00033 * distance_to_go * off_tk_angle) + self.track_bearing

            # radius = (self.sim[prp.airspeed] * 0.5925 / (20.0 * math.pi)) * 1852  # rate 1 radius
            radius = 300
            self.heading_hold(heading)
            if distance < radius:
                self.nav = None

    def orbit_point(self, point, radius):
        if self.orbit_nav is None:
            self.orbit_nav = LocalNavigation(self.sim)
            self.orbit_nav.set_local_target(point[0], point[1])
        if self.orbit_nav is not None:
            bearing = self.orbit_nav.bearing()
            if bearing < 0:
                bearing = bearing + (2 * math.pi)
            circle_bearing = bearing - math.pi
            if circle_bearing < 0:
                circle_bearing = circle_bearing + (2 * math.pi)
            circle_delta_y = radius * math.cos(circle_bearing)
            circle_delta_x = radius * math.sin(circle_bearing)
            # print(circle_delta_y, circle_delta_x)
            circle_point = (point[0] + circle_delta_y, point[1] + circle_delta_x)
            dist_off = radius / 20
            tangent_bearing = circle_bearing - (math.pi / 2)
            distance = self.orbit_nav.distance()
            # error = (distance - radius) / 5
            # heading = tangent_bearing * (180 / math.pi) - error
            # print(heading, self.sim[prp.heading_deg])
            # print(distance)
            offset_delta_y = dist_off * math.cos(tangent_bearing)
            offset_delta_x = dist_off * math.sin(tangent_bearing)
            offset_point = (circle_point[0] + offset_delta_y, circle_point[1] + offset_delta_x)
            # self.orbit_nav.local_target_set = False
            # self.orbit_nav.set_local_target(offset_point[0], offset_point[1])
            leading_bearing = self.orbit_nav.bearing() * 180.0 / math.pi
            if leading_bearing < 0:
                leading_bearing = leading_bearing + 360.0
            tangent_bearing = tangent_bearing * 180.0 / math.pi
            if tangent_bearing < 0:
                tangent_bearing = tangent_bearing + 360.0
            off_tk_angle = leading_bearing - tangent_bearing
            if off_tk_angle < -180.0:
                off_tk_angle = off_tk_angle + 360.0
            if off_tk_angle > 180.0:
                off_tk_angle = off_tk_angle - 360.0
            heading = (2 * off_tk_angle) + tangent_bearing
            if heading > 360.0:
                heading = heading - 360.0
            self.heading_hold(heading)
            print(distance)

    def fillet_path(self, profile, radius):
        #  Not sure why the fillet points can't get round final corner
        if self.nav is None:
            # print(self.state)
            print('Changing points !!!!!!!!!!!!!!!')
            self.track_id = self.track_id + 1
            print(self.track_id)
            if self.track_id == len(profile) - 2:
                print('hit flag')
                self.flag = True
                return self.flag
            point_a = profile[self.track_id]
            point_b = profile[self.track_id + 1]
            point_c = profile[self.track_id + 2]
            # Initialize track inbound to b
            self.nav = LocalNavigation(self.sim)
            self.nav.set_local_target(point_b[0] - point_a[0], point_b[1] - point_a[1])
            self.track_bearing_in = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing_in < 0:
                self.track_bearing_in = self.track_bearing_in + 360.0
            # Initialize track outbound from b
            self.nav.local_target_set = False
            self.nav.set_local_target(point_c[0] - point_b[0], point_c[1] - point_b[1])
            self.track_bearing_out = self.nav.bearing() * 180.0 / math.pi
            if self.track_bearing_out < 0:
                self.track_bearing_out = self.track_bearing_out + 360.0
            self.track_distance = self.nav.distance()
            self.flag = False
            print(self.track_bearing_in)
        if self.nav is not None:
            filet_angle = self.track_bearing_out - self.track_bearing_in
            if self.state == 0:
                r_point = profile[self.track_id]
                q = self.nav.unit_dir_vector(profile[self.track_id], profile[self.track_id + 1])
                w = profile[self.track_id + 1]
                z_point = (w[0] - ((radius / math.tan(filet_angle / 2)) * q[0]),
                           w[1] - ((radius / math.tan(filet_angle / 2)) * q[1]))
                # print(self.track_bearing_in, self.track_bearing_out)
                cur = self.nav.get_local_pos()
                h_point = (cur[0] - z_point[0], cur[1] - z_point[1])
                # print(z_point, cur)
                h_val = (h_point[0] * q[0]) + (h_point[1] * q[1])
                if h_val > 0:
                    # entered h plane
                    self.state = 1
                # Track straight line segment
                self.nav.local_target_set = False  # break target guard
                self.nav.set_local_target(r_point[0], r_point[1])
                bearing = self.nav.bearing() * 180.0 / math.pi
                if bearing < 0:
                    bearing = bearing + 360
                distance = self.nav.distance()
                off_tk_angle = bearing - self.track_bearing_in
                # print(bearing)
                if off_tk_angle > 180:
                    off_tk_angle = off_tk_angle - 360.0
                # scale response with distance from target
                distance_to_go = self.nav.distance_to_go(distance, off_tk_angle)
                if distance_to_go > 3000:
                    distance_to_go = 3000
                heading = (8 * 0.00033 * distance_to_go * off_tk_angle) + self.track_bearing_in
                heading = self.track_bearing_in
                # print(self.track_bearing_in)
                self.heading_hold(heading)
            if self.state == 1:
                q0 = self.nav.unit_dir_vector(profile[self.track_id], profile[self.track_id + 1])
                q1 = self.nav.unit_dir_vector(profile[self.track_id + 1], profile[self.track_id + 2])
                q_grad = self.nav.unit_dir_vector(q1, q0)
                w = profile[self.track_id + 1]
                center_point = (w[0] - ((radius / math.tan(filet_angle / 2)) * q_grad[0]),
                                w[1] - ((radius / math.tan(filet_angle / 2)) * q_grad[1]))
                z_point = (w[0] + ((radius / math.tan(filet_angle / 2)) * q1[0]),
                           w[1] + ((radius / math.tan(filet_angle / 2)) * q1[1]))
                # print('center point:', center_point)
                # print(q1, q0)
                turning_direction = math.copysign(1, (q0[0] * q1[1]) - (q0[1] * q1[0]))
                cur = self.nav.get_local_pos()
                h_point = (cur[0] - z_point[0], cur[1] - z_point[1])
                h_val = (h_point[0] * q1[0]) + (h_point[1] * q1[1])
                if h_val > 0:
                    self.nav = None
                    self.state = 0
                    return

                heading = self.sim[prp.heading_deg]
                distance_from_center = math.sqrt(math.pow(cur[0] - center_point[0], 2) +
                                                 math.pow(cur[1] - center_point[1], 2))
                circ_x = cur[1] - center_point[1]
                circ_y = cur[0] - center_point[0]
                # print(center_point, cur, z_point, heading)
                # print(circ_x, circ_y)
                circle_angle = math.atan2(circ_x, circ_y)
                # if circ_x < 0 < circ_y or circ_y < 0 < circ_x:
                #     circle_angle = circle_angle + math.pi
                if circ_x < 0 and circ_y < 0:
                    circle_angle = circle_angle + (2 * math.pi)
                # print(circle_angle * (180.0 / math.pi))
                tangent_track = circle_angle + (turning_direction * (math.pi / 2))
                if tangent_track < 0:
                    tangent_track = tangent_track + (2 * math.pi)
                if tangent_track > 2 * math.pi:
                    tangent_track = tangent_track - (2 * math.pi)
                tangent_track = tangent_track * (180.0 / math.pi)
                # print(tangent_track)
                heading = tangent_track
                # print(heading)
                # print(self.sim[prp.heading_deg])


                # print(circle_angle * 180 / math.pi)
                # if circle_angle - heading < - math.pi:
                #     circle_angle = circle_angle + 2 * math.pi
                # if circle_angle - heading > math.pi:
                #     circle_angle = circle_angle - 2 * math.pi
                # error = (distance_from_center - radius) / radius
                # k_orbit = 1.0
                # heading = (circle_angle + error_sign * ((math.pi / 2) + math.atan(k_orbit * - error))) * 180 / math.pi
                # print(math.atan(k_orbit * error) * 180 / math.pi)
                self.heading_hold(heading)




















