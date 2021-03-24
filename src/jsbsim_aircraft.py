import collections


"""
Based on https://github.com/Gor-Ren/gym-jsbsim/blob/master/gym_jsbsim/aircraft.py by Gordon Rennie.
Defines allowable aircraft types, this package is based around the x8 light UAV aircraft. 
"""


class Aircraft(collections.namedtuple('Aircraft', ['jsbsim_id', 'name', 'cruise_speed_kts'])):

    KTS_TO_M_PER_S = 0.51444
    KTS_TO_FT_PER_S = 1.6878

    def get_max_distance_m(self, episode_time_s: float) -> float:
        """ Estimate maximum possible distance travelled in an episode """
        margin = 0.1
        return self.cruise_speed_kts * self.KTS_TO_M_PER_S * episode_time_s * (1 + margin)

    def get_cruise_speed_fps(self) -> float:
        return self.cruise_speed_kts * self.KTS_TO_FT_PER_S


cessna172P = Aircraft('c172p', 'Cessna172P', 95)
x8 = Aircraft('x8', 'Skywalker x8', 20)
ball = Aircraft('ball', 'a ball', 5)
# f15 = Aircraft('f15', 'F15', 220)
# a320 = Aircraft('A320', 'Airbus A320', 490)
