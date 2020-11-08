import math
import jsbsim_properties as prp


class Navigation:
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
        a = (math.sin(delta_lat / 2) * math.sin(delta_lat / 2)) +\
            (math.cos(lat_cur_rad) * math.cos(lat_tgt_rad) * math.sin(delta_long / 2) * math.sin(delta_long / 2))
        c = math.atan2(math.sqrt(a), math.sqrt(1-a))
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
        x = (math.cos(lat_cur_rad) * math.sin(lat_tgt_rad)) -\
            math.sin(lat_cur_rad) * math.cos(lat_tgt_rad) * math.cos(delta_long)
        theta = math.atan2(y, x)
        bearing_deg = (theta * (180 / math.pi) + 360) % 360  # use modulo to get value as 360 deg
        return bearing_deg


class LocalNavigation:
    def __init__(self, sim):
        self.sim = sim
        self.tgt = [0, 0]
        self.local_target_set = False

    def set_local_target(self, target_north, target_east):
        if not self.local_target_set:
            cur = self.get_local_pos()
            self.tgt = [target_north + cur[0], target_east + cur[1]]
            self.local_target_set = True
            return self.tgt
        else:
            return self.tgt

    def get_local_pos(self):
        # cur = [self.sim[prp.lat_travel_m], self.sim[prp.lng_travel_m]]
        lat = 111320 * self.sim[prp.lat_geod_deg]
        long = 40075000 * self.sim[prp.lng_geoc_deg] * math.cos(self.sim[prp.lat_geod_deg] * (math.pi / 180.0)) / 360
        cur = (lat, long)
        return cur

    def bearing(self):
        cur = self.get_local_pos()
        y = self.tgt[0] - cur[0]
        x = self.tgt[1] - cur[1]
        # print('y is:', y, 'x is:', x)
        bearing = math.atan2(x, y)
        return bearing

    def distance(self):
        cur = self.get_local_pos()
        y = self.tgt[0] - cur[0]
        x = self.tgt[1] - cur[1]
        # print(self.tgt, cur)
        distance = math.sqrt((x * x) + (y * y))
        return distance

    @staticmethod
    def x_track_error(distance, off_tk_angle):
        x_track = distance * math.sin(off_tk_angle)
        return x_track

    @staticmethod
    def distance_to_go(distance, off_tk_angle):
        distance_to_go = distance * math.sin(off_tk_angle)
        return distance_to_go

    @staticmethod
    def unit_dir_vector(start_point: tuple, end_point: tuple) -> tuple:
        """
        Calculates unitary direction vector between 2 2D points
        :param start_point: where track starts from
        :param end_point: where track ends
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





