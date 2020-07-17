from typing import Union
import numpy as np
from pyproj import Proj, transform

proj_nztm = Proj("epsg:2193")
proj_wgs = Proj("epsg:4326")

class SubductionTile:
    def __init__(self):
        self._centre_nztm, self._centre_lonlat, self._corner_array = (None,) * 3

    @property
    def corner_array(self):
        return self._corner_array

    @corner_array.setter
    def corner_array(self, corner_array: np.ndarray):
        assert corner_array.shape in [(4, 3), (5, 3)], "Expecting 4 corners (rows) and xyz coordinates (3 columns)"
        if corner_array.shape[0] == 5:
            corners = corner_array[:-1, :]

    lon_range_conditions = any([])


