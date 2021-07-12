"""
Script to identify multifault ruptures in Mark's rupture sets
"""

import geopandas as gpd
import numpy as np
from typing import Union
from itertools import product


def find_nearby_vertices(fault1: np.ndarray, fault2: np.ndarray, tolerance: Union[float, int] = 50.):
    """
    Function to find whether two faults (arrays of vertices) overlap, with two or more of the same vertices
    (within a tolerance in m; requires cartesian coordinate system)
    """
    counter = 0
    close_vertices = []
    for vertex in fault1:
        diff_vectors = fault2 - vertex
        differences = np.linalg.norm(diff_vectors, axis=1)
        if differences.min() <= tolerance:
            counter += 1
            close_vertices.append(vertex)

    return counter, close_vertices


def compare_faults(fault1: gpd.GeoDataFrame, fault2: gpd.GeoDataFrame):
    if fault1.NAME != fault2.NAME:
        f1_array = np.array(fault1.geometry.coords)
        f2_array = np.array(fault2.geometry.coords)
        counter, close_vertices = find_nearby_vertices(f1_array, f2_array)

        if counter > 1:
            return [{fault1.NAME, fault2.NAME}, counter, close_vertices]
        else:
            return None
    else:
        return None


if __name__ == "__main__":
    stirling = gpd.read_file("../../../../data/cfm_shapefile/stirling2010.gpkg").explode()
    stirling_nztm = stirling.to_crs(2193)
    stirling_faults = list(stirling_nztm.iterrows())
    overlapping_ruptures = []
    overlapping_rupture_names = []

    for f1, f2 in product(stirling_faults, stirling_faults):
        comparison = compare_faults(f1[1], f2[1])
        if comparison is not None:
            if comparison[0] not in overlapping_rupture_names:
                overlapping_ruptures.append(comparison)
                overlapping_rupture_names.append(comparison[0])






