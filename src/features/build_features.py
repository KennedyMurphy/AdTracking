import numpy as np
from math import pi
from datetime import datetime


def convert_time(dt):
    """ Converts the provided datetime to a tuple that represents
        the coordinates on a clock.

        :param dt:      Datetime object to be converted

        :return:        tuple of clock coordinates
    """

    if not isinstance(dt, datetime):
        raise ValueError("Expected datetime object. Gor class [{}]".format(
            dt.__class__))

    rad = (360 + ((3 - dt.hour) / 12) * 360) * (pi / 180)

    return (np.cos(rad), np.sin(rad))
