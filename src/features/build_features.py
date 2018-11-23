import numpy as np
import pandas as pd
import logging
from math import pi
from datetime import datetime

BASE_FEATURES = ['ip', 'app', 'device', 'os', 'channel']


def feature_creation(data_directory):
    """ Reads in the raw data stored in the data_directory
        (either sample, train, or test) and builds the required 
        features.

        :param data_directory:      Directory of CSV to parse

        :return:    Pandas data frame with features
    """

    logger = logging.getLogger(__name__)
    logger.info("Feature Creation: Reading in data from %s" % data_directory)

    df = pd.read_csv(data_directory)
    assert np.isin(BASE_FEATURES, df.columns).all()

    logger.info("Feature Creation: Generating temporal features")
    assert 'click_time' in df.columns
    # df['click_time'] = pd.to_datetime(df['click_time'])

    # The below feature did not increase the score on the private or publice LB
    # df['day_of_week'] = df.click_time.apply(lambda x: x.dayofweek)

    # Drop temporal columns that are not features
    drop_cols = [c for c in df.columns if "_time" in c]
    df = df.drop(columns=drop_cols)

    return df


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
