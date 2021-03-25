import math
import src.jsbsim_properties as prp
import statistics


class GlobalNavigation:
    # http://www.movable-type.co.uk/scripts/latlong.html was a useful reference
    def __init__(self, sim):
        self.sim = sim
        self.cur = self.set_current_pos()
        self.tgt = self.set_target_pos(0, 0, 0)

    def set_current_pos(self):
        self.cur = [self.sim[prp.lat_geod_deg], self.sim[prp.lng_geoc_deg], self.sim[prp.altitude_sl_ft]]
        return self.cur

    def set_target_pos(self, target_lat, target_long, target_alt):
        self.tgt = [target_lat, target_long, target_alt]
        return self.tgt

    def haversine_distance(self):
        # Using the Haversine distance formula https://en.wikipedia.org/wiki/Haversine_formula
        self.cur = self.set_current_pos()
        earth_rad = 6371e3
        lat_cur_rad = self.cur[0] * (math.pi / 180.0)
        lat_tgt_rad = self.tgt[0] * (math.pi / 180.0)
        long_cur_rad = self.cur[0] * (math.pi / 180.0)
        long_tgt_rad = self.tgt[0] * (math.pi / 180.0)
        delta_lat = lat_tgt_rad - lat_cur_rad
        delta_long = long_tgt_rad - long_cur_rad
        a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2)) + \
            (math.cos(lat_cur_rad) * math.cos(lat_tgt_rad) * math.sin(delta_long / 2) * math.sin(delta_long / 2))
        c = math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_rad * c
        return distance

    def gc_bearing(self):
        # Great circle bearing from current position, also forward azimuth, need to be updated for convergence
        self.cur = self.set_current_pos()
        lat_cur_rad = self.cur[0] * (math.pi / 180.0)
        lat_tgt_rad = self.tgt[0] * (math.pi / 180.0)
        long_cur_rad = self.cur[0] * (math.pi / 180.0)
        long_tgt_rad = self.tgt[0] * (math.pi / 180.0)
        delta_long = long_tgt_rad - long_cur_rad
        y = math.sin(delta_long) * math.cos(lat_tgt_rad)
        x = (math.cos(lat_cur_rad) * math.sin(lat_tgt_rad)) - \
            math.sin(lat_cur_rad) * math.cos(lat_tgt_rad) * math.cos(delta_long)
        theta = math.atan2(y, x)
        bearing_deg = (theta * (180 / math.pi) + 360) % 360  # use modulo to get value as 360 deg
        return bearing_deg


class LocalNavigation:
    """
    Deals with all the local navigation tracking requirements

    ...

    Attributes:
    ----------
    sim : Simulation object
        an instance of the flight simulation flight dynamic model, used to interface with JSBSim
    tgt : list
        the target navigation point
    local_target_set : bool
        acts as a guard to draw a straight track between points

    Methods:
    -------
    set_local_target(target_north, target_east)
        set a target based on the current position
    get_local_pos()
        get the aircraft's current position
    bearing()
        calculate the bearing between the current position and target position
    distance()
        calculate the distance from current position to the target position
    x_track_error(distance, off_tk_angle)
        calculate the distance between the current position and track
    distance_to_go(distance, off_tk_angle)
        calculate the distance to go along the track
    unit_dir_vector(start_point, end_point)
        calculate the unitary direction vector between 2 2D points
    """
    def __init__(self, sim):
        self.sim = sim
        self.tgt: list = [0, 0]
        self.local_target_set: bool = False

    def set_local_target(self, target_north: float, target_east: float) -> list:
        """
        set a target based on the current position

        :param target_north: target latitude (-ve for south)
        :param target_east: target longitude (-ve for west)
        :return: target location
        """
        if not self.local_target_set:
            cur = self.get_local_pos()
            self.tgt = [target_north + cur[0], target_east + cur[1]]
            self.local_target_set = True
            return self.tgt
        else:
            return self.tgt

    def get_local_pos(self) -> tuple:
        """
        Get the aircraft's current position

        :return: current position
        """
        lat = 111320 * self.sim[prp.lat_geod_deg]
        long = 40075000 * self.sim[prp.lng_geoc_deg] * math.cos(self.sim[prp.lat_geod_deg] * (math.pi / 180.0)) / 360
        cur = (lat, long)
        return cur

    def bearing(self) -> float:
        """
        Calculate the bearing between the current position and target position

        :return: bearing [degrees]
        """
        cur = self.get_local_pos()
        y = self.tgt[0] - cur[0]
        x = self.tgt[1] - cur[1]
        # print('y is:', y, 'x is:', x)
        bearing = math.atan2(x, y)
        return bearing

    def distance(self) -> float:
        """
        Calculate the distance from current position to the target position

        :return: distance from target [m]
        """
        cur = self.get_local_pos()
        y = self.tgt[0] - cur[0]
        x = self.tgt[1] - cur[1]
        # print(self.tgt, cur)
        distance = math.sqrt((x * x) + (y * y))
        return distance

    @staticmethod
    def x_track_error(distance: float, off_tk_angle: float) -> float:
        """
        Calculate the distance between the current position and track

        :param distance: distance from the target [m]
        :param off_tk_angle: angle between the track and bearing to target [radians]
        :return: x_track distance [m]
        """
        x_track = distance * math.sin(off_tk_angle)
        return x_track

    @staticmethod
    def distance_to_go(distance: float, off_tk_angle: float) -> float:
        """
        Calculate the distance to go along the track

        :param distance: distance from target [m]
        :param off_tk_angle: angle between the track and bearing to target [radians]
        :return: distance along track to target [m]
        """
        distance_to_go = distance * math.sin(off_tk_angle)
        return distance_to_go

    @staticmethod
    def unit_dir_vector(start_point: tuple, end_point: tuple) -> tuple:
        """
        Calculate the unitary direction vector between 2 2D points

        :param start_point: track starting position
        :param end_point: track ending position
        :return: the normalized direction vector
        """
        direction_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        try:
            unit_vector_n = direction_vector[0] / math.sqrt(math.pow(direction_vector[0], 2)
                                                            + math.pow(direction_vector[1], 2))
        except ZeroDivisionError:
            unit_vector_n = 0
        try:
            unit_vector_e = direction_vector[1] / math.sqrt(math.pow(direction_vector[0], 2)
                                                            + math.pow(direction_vector[1], 2))
        except ZeroDivisionError:
            unit_vector_e = 0

        unit_vector = (unit_vector_n, unit_vector_e)
        return unit_vector


class WindEstimation:
    """
    A class to estimate the wind acting on the vehicle within the environment

    ...

    Attributes:
    ----------
    sim : Simulation object
        an instance of the flight simulation flight dynamic model, used to interface with JSBSim
    cur : tuple
        current position [m]
    old_cur : tuple
        position one timestep behind current timestep
    track_angle : float
        direction of travel from old_cur to cur [radians]
    ground_speed : float
        track_distance_travelled / dt = ground_speed [m/s]
    heading : float
        aircraft heading at old_cur [radians]
    airspeed : float
        aircraft true airspeed [m/s]
    wind_values : list[tuple]
        list containing tuple of wind speed and wind angle
    dt : float
        timestep length [secs], frequency = 1 / dt

    Methods:
    -------
    get_current_pos()
        get the aircraft's current position in lat and long from a [0, 0] center of earth origin
    get_aircraft_state()
        get aircraft simulation variables airspeed and heading
    track()
        get ground_speed and track_angle from old_cur to cur
    wind_components()
        calculate the observed wind speed and wind bearing
    wind_data(n)
        store the wind data calculated from the wind_components to calculate the expected value of wind in subsequent
        time steps
    wind_average(n)
        calculate the arithmetic mean of wind tuples over n observations as a rolling average
    """

    def __init__(self, sim):
        self.sim = sim
        self.cur = (0, 0)
        self.old_cur = (0, 0)
        self.track_angle = 0.0
        self.ground_speed = 0.0
        self.heading = 0.0
        self.airspeed = 0.0
        self.wind_values = []
        self.dt = self.sim[prp.sim_dt]

    def get_current_pos(self) -> tuple:
        """
        Get the aircraft's current position in lat and long from a [0, 0] center of earth origin

        :var: latitude [m]
        :var: longitude [m]

        :return: current position
        """
        lat = 111320 * self.sim[prp.lat_geod_deg]
        long = 40075000 * self.sim[prp.lng_geoc_deg] * math.cos(self.sim[prp.lat_geod_deg] * (math.pi / 180.0)) / 360
        self.cur = (lat, long)
        return self.cur

    def get_aircraft_state(self) -> None:
        """
        Get aircraft simulation variables airspeed and heading

        :var: airspeed [m/s]
        :var: heading [rad]
        :return: None
        """
        self.airspeed = self.sim[prp.airspeed] * 0.3048
        self.heading = self.sim[prp.heading_deg] * (math.pi / 180.0)

    def track(self) -> None:
        """
        Get ground_speed and track_angle from old_cur to cur

        :return:
        """
        self.old_cur = self.cur
        self.get_current_pos()
        track_vector = (self.old_cur[0] - self.cur[0], self.old_cur[1] - self.cur[1])
        self.track_angle = math.atan2(track_vector[1], track_vector[0]) - math.pi
        if self.track_angle < 0:
            self.track_angle = self.track_angle + (2 * math.pi)
        self.ground_speed = math.sqrt(pow(track_vector[0], 2) + pow(track_vector[1], 2)) / self.dt
        print(self.cur, self.old_cur)

    def wind_components(self) -> tuple:
        """
        Calculate the observed wind speed and wind bearing

        :var: wind_speed [m/s]
        :var: wind_angle [rads]
        :return: wind tuple of 2 above variables
        """
        self.track()
        print(self.airspeed, self.ground_speed, self.track_angle * (180 / math.pi), self.heading * (180 / math.pi))
        wind_speed = math.sqrt(pow(self.airspeed, 2) + pow(self.ground_speed, 2) -
                               (2 * self.airspeed * self.ground_speed * math.cos(self.track_angle - self.heading)))
        try:
            wind_angle = math.pi + self.track_angle + \
                         math.asin((self.airspeed * math.sin(self.track_angle - self.heading)) / wind_speed)
        except ZeroDivisionError:
            wind_angle = 0.0
        if wind_angle > 2 * math.pi:
            wind_angle = wind_angle - (2 * math.pi)
        if wind_angle < 0:
            wind_angle = wind_angle + (2 * math.pi)
        self.get_aircraft_state()  # update aircraft state to returned ensuing point
        wind = (wind_speed, wind_angle * (180 / math.pi))
        print(wind)
        return wind

    def wind_data(self, n) -> list:
        """
        Store the wind data calculated from the wind_components to calculate the expected value of wind in subsequent
        time steps

        :param n: number of wind tuples in sample observation
        :return: wind_sample containing n number of observations
        """
        wind = self.wind_components()
        self.wind_values.append(wind)  # is this memory efficient?
        wind_sample = self.wind_values[-n:]  # slice the last n values of wind (default to 1000)
        # print(self.wind_values[-n:])
        return wind_sample

    def wind_average(self, n=1000) -> tuple:
        """
        Calculate the arithmetic mean of wind tuples over n observations as a rolling average

        :param n:
        :return: average wind
        """
        wind_sample = self.wind_data(n)
        try:
            arithmetic_mean_speed = statistics.mean(wind_sample[0])
            arithmetic_mean_direction = statistics.mean(wind_sample[1])
            arithmetic_mean_wind = (arithmetic_mean_speed, arithmetic_mean_direction)
            return arithmetic_mean_wind
        except IndexError:
            arithmetic_mean_wind = (0, 0)
        # print(arithmetic_mean_wind)
