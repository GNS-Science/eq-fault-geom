import geopandas as gpd
import pandas as pd
import numpy as np

# Dictionary of direction vectors.
pi4 = 0.25*np.pi
direction_vecs = {"N": np.array([ 0.0,  1.0,  0.0], dtype=np.float64),
                  "E": np.array([ 1.0,  0.0,  0.0], dtype=np.float64),
                  "S": np.array([ 0.0, -1.0,  0.0], dtype=np.float64),
                  "W": np.array([-1.0,  0.0,  0.0], dtype=np.float64),
                  "NE": np.array([np.cos(pi4),  np.sin(pi4),  0.0], dtype=np.float64),
                  "SE": np.array([np.cos(-pi4),  np.sin(-pi4),  0.0], dtype=np.float64),
                  "SW": np.array([np.cos(5.0*pi4),  np.sin(5.0*pi4),  0.0], dtype=np.float64),
                  "NW": np.array([np.cos(3.0*pi4),  np.sin(3.0*pi4),  0.0], dtype=np.float64)}


def calculate_dip_rotation(fault_info: pd.Series):
    """
    Calculate slope of fault trace in NZTM, then add 90 to get dip direction.
    Form 3D rotation matrix from dip direction.
    :param fault_info: Pandas series
    :return:
    """
    # Get dip direction from fault_info.
    dip_dir = fault_info["Dip_Dir"]

    # Get coordinates
    x, y = fault_info.geometry.xy

    # Calculate gradient of line in 2D
    p = np.polyfit(x, y, 1)
    gradient = p[0]

    # Gradient to normal
    normal = np.arctan2(gradient, 1) - 0.5*np.pi
    normal_dir = np.array([np.cos(normal), np.sin(normal), 0.0], dtype=np.float64)

    # Test dip direction against direction vector.
    if (dip_dir != None):
        test_dot = np.dot(normal_dir, direction_vecs[dip_dir])
        if (test_dot < 0.0):
            normal += np.pi

    # Rotation matrix.
    cosn = np.cos(normal)
    sinn = np.sin(normal)
    rot_mat = np.array([[cosn, -sinn, 0.0],
                        [sinn, cosn, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

    return rot_mat


