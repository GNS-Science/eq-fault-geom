from typing import Union

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString

dip_direction_ranges = {"E": (45, 135), "NE": (0, 90), "N": (315, 45), "NW": (270, 360), "SW": (180, 270),
                        "S": (135, 225), "SE": (90, 180), "W": (225, 315)}
valid_dip_directions = list(dip_direction_ranges.keys()) + [None]

dominant_rake_ranges = {"reverse": (225, 315), "dextral": (325, 45), "sinistral": (135, 315), "normal": (45, 135)}
secondary_rake_ranges = {"reverse": (180, 360), "dextral": (270, 90), "sinistral": (90, 270), "normal": (0, 180)}
possible_rake_dirs = ['dextral', 'normal', 'reverse', 'sinistral']

valid_dip_range = [0, 90]
valid_depth_range = [0, 50]
valid_rake_range = [0, 360]
valid_sr_range = [0, 50]

expected_fields = ['Depth_Best', 'Depth_Max', 'Depth_Min', 'Dip_Best',
                   'Dip_Dir', 'Dip_Max', 'Dip_Min', 'FZ_Name',
                   'Qual_Code', 'Rake_Best', 'Rake_Max', 'Rake_Min', 'Sense_Dom',
                   'Sense_Sec', 'Source1_1', 'Source2', 'SR_Best', 'SR_Max', 'SR_Min',
                   'geometry']

required_fields = ['Name', 'Number', 'geometry']


def reverse_bearing(bearing: Union[int, float]):
    assert isinstance(bearing, (float, int))
    assert 0. <= bearing <= 360.
    new_bearing = bearing + 180.

    # Ensure strike is between zero and 360 (bearing)
    while new_bearing < 0:
        new_bearing += 360.

    while new_bearing >= 360.:
        new_bearing -= 360.

    return new_bearing


def reverse_line(line: LineString):
    assert isinstance(line, LineString)
    x, y = line.xy
    x_back = x[-1::-1]
    y_back = y[-1::-1]
    new_line = LineString([[xi, yi] for xi, yi in zip(x_back, y_back)])
    return new_line


def calculate_dip_direction(line: LineString):
    """
    Calculate the strike of a shapely linestring object with coordinates in NZTM,
    then adds 90 to get dip direction.
    :param line: Linestring object
    :return:
    """
    # Get coordinates
    x, y = line.xy
    # Calculate gradient of line in 2D
    p = np.polyfit(x, y, 1)
    gradient = p[0]
    # Gradient to bearing
    bearing = 180 - np.degrees(np.arctan2(gradient, 1))
    bearing_vector = np.array([np.sin(np.radians(bearing)), np.cos(np.radians(bearing))])

    # Determine whether line object fits strike convention
    relative_x = x - x[0]
    relative_y = y - y[0]

    distances = np.matmul(np.vstack((relative_x, relative_y)), bearing_vector)
    num_pos = len(distances >= 0)
    num_neg = len(distances < 0)

    if num_neg > num_pos:
        bearing += 180.
        line_rh_convention = reverse_line(line)
    else:
        line_rh_convention = line

    # Ensure strike is between zero and 360 (bearing)
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing, line_rh_convention


def root_mean_square(value_array: Union[np.ndarray, list, tuple]):
    """
    Helper function to turn max and min to stdev for inclusion in XML.
    :param value_array: Differences of values (e.g. sr_min and sr_max) from mean.
    :return:
    """
    data_array = np.array(value_array)
    assert all([data_array.size > 0, data_array.ndim == 1])
    rms = np.sqrt(np.mean(np.square(data_array)))
    return rms


class CfmMultiFault:
    def __init__(self, fault_geodataframe: gpd.GeoDataFrame):

        for field in required_fields:
            if field not in fault_geodataframe.columns:
                raise ValueError("Missing required field: {}".format(field))
        for field in expected_fields:
            if field not in fault_geodataframe.columns:
                print("Warning: missing expected field: {}".format(field))

        self._faults = []

        for i, row in fault_geodataframe.iterrows():
            self.add_fault(row)

    @property
    def faults(self):
        return self._faults

    def add_fault(self, series: pd.Series):
        self.faults.append(CfmFault.from_series(series, parent_multifault=self))

    @property
    def fault_numbers(self):
        if self.faults is not None:
            return [fault.number for fault in self.faults]
        else:
            return []



class CfmFault:
    def __init__(self, parent_multifault: CfmMultiFault = None):
        # Attributes usually provided in CFM trace shapefile
        self._parent = parent_multifault
        self._depth_best, self._depth_min, self._depth_max = (None,) * 3
        self._dip_best, self._dip_min, self._dip_max, self._dip_dir, self._dip_dir_str = (None,) * 5
        self._fz_name, self._name = (None,) * 2
        self._number, self._qual_code = (None,) * 2
        self._rake_best, self._rake_max, self._rake_min = (None,) * 3
        self._sense_dom, self._sense_sec = (None,) * 2
        self._source1_1, self.source2 = (None,) * 2
        self._sr_best, self._sr_max, self._sr_min = (None,) * 3
        self._nztm_trace = None

        # Attributes required for OpenSHA XML
        self._section_id, self._section_name = (None,) * 2
        self._dip_sigma, self._rake_sigma, self._sr_sigma = (None,) * 3

    # Depths
    @property
    def depth_best(self):
        return self._depth_best

    @property
    def depth_max(self):
        return self._depth_max

    @property
    def depth_min(self):
        return self._depth_min

    @depth_best.setter
    def depth_best(self, depth: Union[float, int]):
        depth_v = self.validate_depth(depth)
        if self.depth_min is not None:
            if depth_v < self.depth_min:
                print("Warning: depth_best lower than depth_min ({})".format(self.name))
        if self.depth_max is not None:
            if depth_v > self.depth_max:
                print("Warning: depth_best greater than depth_max ({})".format(self.name))
        self._depth_best = depth_v

    @depth_max.setter
    def depth_max(self, depth: Union[float, int]):
        depth_v = self.validate_depth(depth)
        for depth_value in (self.depth_min, self.depth_best):
            if depth_value is not None and depth_v < depth_value:
                print("Warning: depth_max lower than either depth_min or depth_best ({})".format(self.name))
        self._depth_max = depth_v

    @depth_min.setter
    def depth_min(self, depth: Union[float, int]):
        depth_v = self.validate_depth(depth)
        for depth_value in (self.depth_max, self.depth_best):
            if depth_value is not None and depth_v > depth_value:
                print("Warning: depth_min higher than either depth_max or depth_best ({})".format(self.name))
        self._depth_max = depth_v

    @staticmethod
    def validate_depth(depth: Union[float, int]):
        assert isinstance(depth, (float, int))
        depth_positive = depth if depth >= 0 else depth * -1
        assert valid_depth_range[0] <= depth_positive <= valid_depth_range[1]
        return depth_positive

    # Dips
    @property
    def dip_best(self):
        return self._dip_best

    @property
    def dip_max(self):
        return self._dip_max

    @property
    def dip_min(self):
        return self._dip_min

    @property
    def dip_dir_str(self):
        return self._dip_dir

    @dip_best.setter
    def dip_best(self, dip: Union[float, int]):
        dip_v = self.validate_dip(dip)
        if self.dip_min is not None:
            if dip_v < self.dip_min:
                print("Warning: dip_best lower than dip_min ({})".format(self.name))
        if self.dip_max is not None:
            if dip_v > self.dip_max:
                print("Warning: dip_best greater than dip_max ({})".format(self.name))
        self._dip_best = dip_v

    @dip_max.setter
    def dip_max(self, dip: Union[float, int]):
        dip_v = self.validate_dip(dip)
        for dip_value in (self.dip_min, self.dip_best):
            if dip_value is not None and dip_v < dip_value:
                print("Warning: dip_max lower than either dip_min or dip_best ({})".format(self.name))
        self._dip_max = dip_v

    @dip_min.setter
    def dip_min(self, dip: Union[float, int]):
        dip_v = self.validate_dip(dip)
        for dip_value in (self.dip_max, self.dip_best):
            if dip_value is not None and dip_v > dip_value:
                print("Warning: dip_min higher than either dip_max or dip_best ({})".format(self.name))
        self._dip_max = dip_v

    @property
    def dip_sigma(self):
        if self._dip_sigma is not None:
            return self._dip_sigma
        elif not any([a is None for a in (self.dip_min, self.dip_max)]):
            return root_mean_square(np.array([self.dip_min, self.dip_max]))
        else:
            raise ValueError("Insufficient data to calculate dip_sigma!")

    @dip_dir_str.setter
    def dip_dir_str(self, dip_dir: str):
        assert any([isinstance(dip_dir, str), dip_dir is None])
        if isinstance(dip_dir, str):
            assert dip_dir.upper() in valid_dip_directions
            self._dip_dir = dip_dir.upper()
        else:
            self._dip_dir = None
        if self.nztm_trace is not None:
            self.validate_dip_direction()

    def validate_dip_direction(self):
        if any([a is None for a in [self.dip_dir_str, self.nztm_trace]]):
            print("Insufficient information to validate dip direction")
            return
        else:
            dd_from_trace, line = calculate_dip_direction(self.nztm_trace)
            min_dd_range, max_dd_range = dip_direction_ranges[self.dip_dir_str]
            if not all([min_dd_range <= dd_from_trace, dd_from_trace <= max_dd_range]):
                reversed_dd = reverse_bearing(dd_from_trace)
                if all([min_dd_range <= reversed_dd, reversed_dd <= max_dd_range]):
                    self._nztm_trace = reverse_line(line)
                    self._dip_dir = reversed_dd
                else:
                    if self.name:
                        print(self.name)
                    print("Supplied trace and dip direction {} are inconsistent: "
                          "please check...".format(self.dip_dir_str))

    @staticmethod
    def validate_dip(dip: Union[float, int]):
        assert isinstance(dip, (float, int))
        assert valid_dip_range[0] <= dip <= valid_dip_range[1]
        return dip

    # Trace
    @property
    def nztm_trace(self):
        return self._nztm_trace

    @nztm_trace.setter
    def nztm_trace(self, trace: LineString):
        assert isinstance(trace, LineString)
        self._nztm_trace = trace

    # Rake and sense of slip
    @property
    def rake_best(self):
        return self._rake_best

    @property
    def rake_max(self):
        return self._rake_max

    @property
    def rake_min(self):
        return self._rake_min

    @property
    def sense_dom(self):
        return self._sense_dom

    @property
    def sense_sec(self):
        return self._sense_sec

    @rake_best.setter
    def rake_best(self, rake: Union[float, int]):
        rake_v = self.validate_rake(rake)
        if self.rake_min is not None:
            if rake_v < self.rake_min:
                print("Warning: rake_best lower than rake_min ({})".format(self.name))
        if self.rake_max is not None:
            if rake_v > self.rake_max:
                print("Warning: rake_best greater than rake_max ({})".format(self.name))
        self._rake_best = rake_v
        if self.sense_dom is not None:
            self.validate_rake_sense()

    @rake_max.setter
    def rake_max(self, rake: Union[float, int]):
        rake_v = self.validate_rake(rake)
        for rake_value in (self.rake_min, self.rake_best):
            if rake_value is not None and rake_v < rake_value:
                print("Warning: rake_max lower than either rake_min or rake_best ({})".format(self.name))
        self._rake_max = rake_v

    @rake_min.setter
    def rake_min(self, rake: Union[float, int]):
        rake_v = self.validate_rake(rake)
        for rake_value in (self.rake_max, self.rake_best):
            if rake_value is not None and rake_v > rake_value:
                print("Warning: rake_min higher than either rake_max or rake_best ({})".format(self.name))
        self._rake_max = rake_v

    @staticmethod
    def validate_rake(rake: Union[float, int]):
        assert isinstance(rake, (float, int))
        if -180 <= rake <= 0:
            rake += 360
        assert valid_rake_range[0] <= rake <= valid_rake_range[1]
        return rake

    @sense_dom.setter
    def sense_dom(self, sense: str):
        assert isinstance(sense, str)
        assert sense.lower() in possible_rake_dirs
        self._sense_dom = sense

    @sense_sec.setter
    def sense_sec(self, sense: str):
        assert any([sense is None, isinstance(sense, str)])
        if sense is not None:
            assert sense.lower() in possible_rake_dirs
        self._sense_sec = sense
        if self.rake_best is not None:
            self.validate_rake_sense()

    def validate_rake_sense(self):
        if any([a is None for a in (self.rake_best, self.sense_dom)]):
            print("{}: Insufficient data to compare rake and slip sense".format(self.name))
            return
        else:
            dominant_range = dominant_rake_ranges[self.sense_dom]
            if not all([self.rake_best >= dominant_range[0],
                        self.rake_best <= dominant_range[1]]):
                print("{}: Supplied rake ({:.2f} deg} differs from dominant slip sense ({})".format(self.name,
                                                                                                    self.rake_best,
                                                                                                    self.sense_dom))
            if self.sense_sec is not None:
                sec_range = secondary_rake_ranges[self.sense_sec]
                if not all([self.rake_best >= sec_range[0],
                            self.rake_best <= sec_range[1]]):
                    print("{}: Supplied rake ({:.2f} deg} inconsistent with sec slip sense ({})".format(self.name,
                                                                                                        self.rake_best,
                                                                                                        self.sense_sec))

    @property
    def sr_best(self):
        return self._sr_best

    @property
    def sr_min(self):
        return self._sr_min

    @property
    def sr_max(self):
        return self._sr_max

    @sr_best.setter
    def sr_best(self, slip_rate: Union[float, int]):
        self.validate_sr(slip_rate)
        if self.sr_min is not None:
            if slip_rate < self.sr_min:
                print("Warning: sr_best lower than sr_min ({})".format(self.name))
        if self.sr_max is not None:
            if slip_rate > self.sr_max:
                print("Warning: sr_best greater than sr_max ({})".format(self.name))
        self._sr_best = slip_rate

    @sr_min.setter
    def sr_min(self, slip_rate: Union[float, int]):
        self.validate_sr(slip_rate)
        if self.sr_best is not None:
            if slip_rate > self.sr_best:
                print("Warning: sr_best lower than sr_min ({})".format(self.name))
        if self.sr_max is not None:
            if slip_rate > self.sr_max:
                print("Warning: sr_min greater than sr_max ({})".format(self.name))
        self._sr_best = slip_rate

    @sr_max.setter
    def sr_max(self, slip_rate: Union[float, int]):
        self.validate_sr(slip_rate)
        if self.sr_min is not None:
            if slip_rate < self.sr_min:
                print("Warning: sr_max lower than sr_min ({})".format(self.name))
        if self.sr_best is not None:
            if slip_rate < self.sr_best:
                print("Warning: sr_best greater than sr_max ({})".format(self.name))
        self._sr_best = slip_rate

    @staticmethod
    def validate_sr(slip_rate: Union[float, int]):
        assert isinstance(slip_rate, (float, int))
        assert valid_sr_range[0] <= slip_rate <= valid_sr_range[1]
        return slip_rate

    @property
    def sr_sigma(self):
        if self._sr_sigma is not None:
            return self._sr_sigma
        elif not any([a is None for a in (self.sr_min, self.sr_max)]):
            return root_mean_square(np.array([self.sr_min, self.sr_max]))
        else:
            raise ValueError("Insufficient data to calculate sr_sigma!")

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name_str: str):
        assert isinstance(name_str, str)
        if not name_str:
            print("Warning: empty name")
        self._name = name_str

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, fault_number):
        assert isinstance(fault_number, int)
        if self.parent is not None:
            if fault_number in self.parent.fault_numbers:
                print("Duplicate fault number: {:d}".format(fault_number))

    @property
    def parent(self):
        return self._parent




    @classmethod
    def from_series(cls, series: pd.Series, parent_multifault: CfmMultiFault = None):

        fault = cls(parent_multifault=parent_multifault)
        fault.name = series["Name"]
        fault.number = series["Number"]

