import geopandas as gpd
import numpy as np


def find_nearest_fault_to_point(point_i: np.ndarray, fault_vertices: np.ndarray, fault_dic: dict):
    """

    """
    diffs = point_i - fault_vertices
    dists = np.linalg.norm(diffs, axis=1)
    loc_min = np.argmin(dists)
    return fault_dic[tuple(fault_vertices[loc_min])]


if __name__ == "__main__":
    cfm = gpd.read_file("../../../data/cfm_shapefile/cfm_0_9.gpkg").explode()
    stirling = gpd.read_file("../../../data/cfm_shapefile/stirling2010.gpkg").explode()

    stirling_nztm = stirling.to_crs(epsg=2193)


    stirling_dic = {}
    stirling_name_dic = {}
    stirling_list = []
    for i, fault in stirling_nztm.iterrows():
        for point in fault.geometry.coords:
            stirling_dic[point] = fault.NAME
            stirling_list.append(point)

    stirling_array = np.array(stirling_list)

    for i, fault in cfm.iterrows():
        closest_list = []
        for point in fault.geometry.coords:

            nearest_fault = find_nearest_fault_to_point(np.array(point), stirling_array, stirling_dic)
            closest_list.append(float(stirling[stirling.NAME == nearest_fault].Depth))
        closest_array = np.array(closest_list)
        # print(closest_array)
        values, counts = np.unique(closest_array, return_counts=True)

        if all([a in fault.Name for a in ("Alpine", "George")]):
            depth = 12.0
        else:
            if len(counts) == 1:
                depth = values[0]
            else:
                counts_desc = np.sort(np.array(counts))[-1::-1]
                if counts_desc[0] > counts_desc[1]:
                    count = counts_desc[0]
                    depth = values[np.argwhere(counts == count)][0][0]
                else:
                    depth = max([value for value in values if value < 35])

        cfm.loc[cfm.Name == fault.Name, "D90"] = depth

    cfm.to_file("cfm_0_9_stirling_depths.shp")




