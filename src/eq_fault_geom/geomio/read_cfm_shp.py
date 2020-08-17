import geopandas as gpd
import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shapely.geometry import LineString
from typing import Union
import numpy as np


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

    # Needs to be improved... Basic method of defining quadrant
    if x[0] > x[-1]:
        bearing += 180.

    # Ensure strike is between zero and 360 (bearing)
    while bearing < 0:
        bearing += 360.

    while bearing >= 360.:
        bearing -= 360.

    return bearing


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


# Currently does nothing... Could be used to implement checks on parameters
required_values = ['Depth_Best', 'Depth_Max', 'Depth_Min', 'Dip_Best',
                   'Dip_Dir', 'Dip_Max', 'Dip_Min', 'FZ_Name', 'Name', 'Number',
                   'Qual_Code', 'Rake_Best', 'Rake_Max', 'Rake_Min', 'Sense_Dom',
                   'Sense_Sec', 'Source1_1', 'Source2', 'SR_Best', 'SR_Max', 'SR_Min',
                   'geometry']


def fault_model_xml(fault_info: pd.Series, section_id: int, nztm_geometry: LineString):
    """
    To generate XML element containing fault metadata from shapefile
    :param fault_info:
    :param section_id:
    :param nztm_geometry:
    :return:
    """
    # Calculate uncertainty on slip rate and dip direction
    sr_stdev = root_mean_square(np.array([fault_info["SR_Max"], fault_info["SR_Min"]]) - fault_info["SR_Best"])
    dip_direction = calculate_dip_direction(nztm_geometry)

    # Unique fault identifier
    tag_name = "i{:d}".format(section_id)
    # Metadata
    attribute_dic = {"sectionId": "{:d}".format(section_id),
                     "sectionName": fault_info.Name,
                     "aveLongTermSlipRate": "{:.1f}".format(fault_info["SR_Best"]),
                     "slipRateStDev": "{:.1f}".format(sr_stdev),
                     "aveDip": "{:.1f}".format(fault_info["Dip_Best"]),
                     "aveRake": "{:.1f}".format(fault_info["Rake_Best"]),
                     "aveUpperDepth": "0.0",
                     "aveLowerDepth": "{:.1f}".format(fault_info["Depth_Best"]),
                     "aseismicSlipFactor": "0.0",
                     "couplingCoeff": "1.0",
                     "dipDirection": "{:.1f}".format(dip_direction),
                     "parentSectionId": "-1",
                     "connector": "false"
                     }
    # Initialize XML element
    fault_element = ET.Element(tag_name, attrib=attribute_dic)
    # Add sub element for fault trace
    trace_element = fault_trace_xml(fault_info.geometry, fault_info.Name)
    fault_element.append(trace_element)
    return fault_element


def fault_trace_xml(geometry: LineString, section_name: str, z: Union[float, int] = 0):
    trace_element = ET.Element("FaultTrace", attrib={"name": section_name})
    ll_float_str = "{:.4f}"
    # extract arrays of lon and lat
    x, y = geometry.xy
    # Loop through addis each coordinate as sub element
    for x_i, y_i in zip(x, y):
        loc_element = ET.Element("Location", attrib={"Latitude": ll_float_str.format(y_i),
                                                     "Longitude": ll_float_str.format(x_i),
                                                     "Depth": ll_float_str.format(z)})
        trace_element.append(loc_element)

    return trace_element


# Example file; should work on whole dataset too
shp_file = "/Users/arh79/PycharmProjects/eq-fault-geom/data/cfm_shapefile/cfm_lower_n_island.shp"

# read in data
shp_df = gpd.GeoDataFrame.from_file(shp_file)
# Sort alphabetically by name
sorted_df = shp_df.sort_values("Name")
# Reset index to line up with alphabetical sorting
sorted_df = sorted_df.reset_index(drop=True)
# Reproject traces into lon lat
sorted_wgs = sorted_df.to_crs(epsg=4326)

# Base XML element
opensha_element = ET.Element("OpenSHA")
# Fault model sub element
fm_element = ET.Element("FaultModel")
opensha_element.append(fm_element)

# Loop through faults, creating XML elements
for i, fault in sorted_wgs.iterrows():
    # Extract NZTM line for dip direction calculation/could be done in a better way, I'm sure
    nztm_geometry_i = sorted_df.iloc[i].geometry
    # Add to XML tree
    opensha_element.append(fault_model_xml(fault, section_id=i, nztm_geometry=nztm_geometry_i))

# Awkward way of getting the xml file to be written in a way that's easy to read.
xml_dom = minidom.parseString(ET.tostring(opensha_element, encoding="UTF-8", xml_declaration=True))
pretty_xml_str = xml_dom.toprettyxml(indent="  ", encoding="utf-8")

# Write output to file
with open("test2.xml", "wb") as fid:
    fid.write(pretty_xml_str)
