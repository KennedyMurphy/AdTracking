import unittest
import numpy as np
from datetime import datetime
from src.features.build_features import convert_time


class TestConvertTime(unittest.TestCase):

    def setUp(self):
        self.t1 = datetime.strptime("2015-01-01 12:00:00", "%Y-%m-%d %H:%M:%S")
        self.t2 = datetime.strptime("2015-01-01 3:00:00", "%Y-%m-%d %H:%M:%S")
        self.t3 = datetime.strptime("2015-01-01 6:00:00", "%Y-%m-%d %H:%M:%S")
        self.t4 = datetime.strptime("2015-01-01 9:00:00", "%Y-%m-%d %H:%M:%S")

        self.t5 = datetime.strptime("2015-01-01 15:00:00", "%Y-%m-%d %H:%M:%S")
        self.t6 = datetime.strptime("2015-01-01 18:00:00", "%Y-%m-%d %H:%M:%S")
        self.t7 = datetime.strptime("2015-01-01 21:00:00", "%Y-%m-%d %H:%M:%S")

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            convert_time("notdatetime")

    def test_result(self):
        self.assertTrue(
            np.equal(np.round(convert_time(self.t1), 0), np.array([0,
                                                                   1])).all())
        self.assertTrue(
            np.equal(np.round(convert_time(self.t2), 0), np.array([1,
                                                                   0])).all())
        self.assertTrue(
            np.equal(np.round(convert_time(self.t3), 0), np.array([0,
                                                                   -1])).all())
        self.assertTrue(
            np.equal(np.round(convert_time(self.t4), 0), np.array([-1,
                                                                   0])).all())

        self.assertTrue(
            np.equal(
                np.round(convert_time(self.t2), 0),
                np.round(convert_time(self.t5), 0)).all())
        self.assertTrue(
            np.equal(
                np.round(convert_time(self.t3), 0),
                np.round(convert_time(self.t6), 0)).all())
        self.assertTrue(
            np.equal(
                np.round(convert_time(self.t4), 0),
                np.round(convert_time(self.t7), 0)).all())


if __name__ == '__main__':
    unittest.main()
