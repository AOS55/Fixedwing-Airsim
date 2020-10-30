from simple_pid import PID
import jsbsim_properties as prp
from jsbsim_simulator import Simulation
from scipy import interpolate
import math


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
        kp = 0.228
        ki = 0.145
        kd = 0.036
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
        kp = 0.10
        ki = 0.00
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

    def vs_hold_w_throttle(self, vs_comm):
        error = vs_comm + self.sim[prp.v_down_fps]
        kp = 0.0
        ki = 0.0
        kd = 0.0
        vs_controller = PID(kp, ki, kd)
        output = vs_controller(error)
        self.sim[prp.throttle_cmd] = output











