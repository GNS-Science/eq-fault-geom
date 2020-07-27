from eq_fault_geom.geomio.array_operations import read_tiff
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import os
import pandas as pd

# Location of data directory: TODO need to decide whether data are installed with project
# data_dir = "/Users/arh79/PycharmProjects/eq-fault-geom/data/"
data_dir = os.path.expanduser("~/DEV/GNS/eq-fault-geom/data/")
output_dir = os.getcwd()
"""
Script to turn a gridded GeoTiff (in NZTM format) of the Hikurangi subduction interface into square tiles
Workflow:
1. Create across-strike profiles, evenly spaced along the strike along strike of the subduction zone
2. Interpolate points along the paths defined by profiles, so that tile centres are evenly spaced within 
   the curved surface of the interface.
3. Find the local strike and dip of the surface at each tile centre, and create a tile with that strike/dip 
   and a prescribed width.
"""


def fit_plane_svd(point_cloud: np.ndarray):
    """
    Fit a plane through points using numpy SVD
    :param point_cloud: (n x 3) numpy array
    :return: unit vector of normal to plane
    """
    # Find centre of point cloud
    g = point_cloud.sum(axis=0) / point_cloud.shape[0]

    # run SVD
    u, s, vh = np.linalg.svd(point_cloud - g)

    # unitary normal vector
    u_norm = vh[2, :]
    return u_norm

"""
Read and pre process data from files, set parameters for meshing
"""

# Relevant quantities that control tile distribution
# Swath profile half width; for making (slightly) smoothed profiles/cross-sections through interface.
profile_half_width = 2000
# Spacing between down-dip profiles (and therefore tiles) in the along-strike direction
profile_spacing = 10000
# Max distance to select points to fit
search_radius = 1e4


# Read in grid from subduction interface
tiff = os.path.join(data_dir, "subduction/williams_0_005_nztm.tif")
# print(data_dir, tiff, Path(tiff).resolve())
x, y, z = read_tiff(tiff)
# Multiply z coordinates by 1000 so that everything is in metres
z *= 1000

# Turn grid into xyz (n x 3) array, by making x and y grids and flattening them.
x_grid, y_grid = np.meshgrid(x, y)
all_xyz_with_nans = np.vstack((x_grid.flatten(), y_grid.flatten(), z.flatten())).T

# Remove points where z is NaN
all_xyz = all_xyz_with_nans[~np.isnan(all_xyz_with_nans).any(axis=1)]

# Read shapefile: line that gives overall strike of subduction zone
# Included so that easy to fiddle with in GIS
overall_trace = gpd.GeoDataFrame.from_file(os.path.join(data_dir, "subduction/overall_trace.shp"))
overall_line = overall_trace.geometry[0]
# Get SE corner of line
corner = np.array(overall_line.coords[0])
# In case (for example) geopandas is not installed
# corner = np.array([1627783.3117604, 4942542.56366084])
# Direction of trace in vector form
overall_vec = np.array(overall_line.coords[1]) - corner
# Turn into degrees for inspection of values
overall_strike = 90 - np.degrees(np.arctan2(overall_vec[1], overall_vec[0]))
# Unit vector along strike
along_overall = overall_vec / np.linalg.norm(overall_vec)
# Rotate to give unit vector perpendicular to strike
across_vec = np.matmul(np.array([[0, -1], [1, 0]]), along_overall)

"""
Calculate locations of tile centres
"""

# Calculate distances of coordinates in mesh from corner point
along_dists = (x_grid - corner[0]) * along_overall[0] + (y_grid - corner[1]) * along_overall[1]
across_dists = (x_grid - corner[0]) * across_vec[0] + (y_grid - corner[1]) * across_vec[1]

# Find start and end locations along strike (when z values stop being NaNs and start again)
start_along = min(along_dists[~np.isnan(z)])
end_along = max(along_dists[~np.isnan(z)])

# Space profiles evenly along strike (distance set by profile spacing variable)
along_spaced = np.arange(start_along + profile_spacing/2, end_along, profile_spacing)

all_points_ls = []

all_indices = []

# Loop through, taking profiles in down-dip direction
for along_index, along in enumerate(along_spaced):
    # Point at end of profile at trench (SE) end
    row_end = corner + along * along_overall

    # Boundaries to select projected points
    along_min = along - profile_half_width
    along_max = along + profile_half_width

    # Extract z values and distances across strike
    in_swath = np.logical_and(along_dists >= along_min, along_dists <= along_max)
    swath_across = across_dists[in_swath]
    swath_z = z[in_swath]

    # Remove nans
    across_no_nans = swath_across[~np.isnan(swath_z)]

    # Start and end of profile after nans have been removed
    start_across = min(across_no_nans)
    end_across = max(across_no_nans)

    # Dummy profile distances to create interpolated profile.
    # Not used for down-dip distances.
    # every 2 km, for now
    initial_spacing = np.arange(start_across, end_across, profile_half_width)

    # Combine and sort distances along profiles (with z)
    across_vs_z = np.vstack((across_no_nans, swath_z[~np.isnan(swath_z)])).T
    sorted_coords = across_vs_z[across_vs_z[:, 0].argsort()]

    # Interpolate, then turn into shapely linestring
    interp_z = np.interp(initial_spacing, sorted_coords[:, 0], sorted_coords[:, 1])
    interp_line = LineString(np.vstack((initial_spacing, interp_z)).T)

    # Interpolate locations of profile centres
    interpolation_distances = np.arange(profile_spacing/2, interp_line.length, profile_spacing)
    interpolated_points = [interp_line.interpolate(distance) for distance in interpolation_distances]

    # Turn coordinates of interpolated points back into arrays
    interpolated_x = np.array([point.x for point in interpolated_points])
    interpolated_z_values = np.array([point.y for point in interpolated_points])

    # Calculate NZTM coordinates of tile centres
    point_xys = np.array([row_end + across_i * across_vec for across_i in interpolated_x])
    point_xyz = np.vstack((point_xys.T, interpolated_z_values)).T

    patch_indices = [(along_index, across_index) for across_index in range(point_xyz.shape[0])]

    # Store in list
    all_points_ls.append(point_xyz)
    all_indices += patch_indices


# List to array
all_points_array = np.vstack(all_points_ls)

"""
Loop through tile centres, fitting planes through nearby points
"""
# Holder for tile polygons
all_tile_ls = []
# Lists to hold info in alternative format
top_traces = []
dips = []
top_depths = []
bottom_depths = []

for centre_point in all_points_array[:5000]:
    # Find distances of all points from centre
    difference_vectors = all_xyz - centre_point
    distances = np.linalg.norm(difference_vectors, axis=1)
    # Select relevant points
    small_cloud = all_xyz[distances < search_radius]

    # Normal to plane
    normal_i = fit_plane_svd(small_cloud)

    # Make sure normal points up
    if normal_i[-1] < 0:
        normal_i *= -1

    # Calculate along-strike vector (left-hand-rule)
    strike_vector = np.cross(normal_i, np.array([0, 0, -1]))
    strike_vector[-1] = 0
    strike_vector /= np.linalg.norm(strike_vector)

    # Create down-dip vector
    down_dip_vector = np.cross(normal_i, strike_vector)
    if down_dip_vector[-1] > 0:
        down_dip_vector *= -1

    dip = np.degrees(np.arctan(-1 * down_dip_vector[-1] / np.linalg.norm(down_dip_vector[:-1])))
    dips.append(dip)

    poly_ls = []
    for i, j in zip([1, 1, -1, -1], [1, -1, -1, 1]):
        corner_i = centre_point + (i * strike_vector + j * down_dip_vector) * profile_spacing / 2
        poly_ls.append(corner_i)

    top_depths.append(poly_ls[1][-1])
    bottom_depths.append(poly_ls[0][-1])

    top_trace = LineString(poly_ls[1:-1])
    top_traces.append(top_trace)



    all_tile_ls.append(np.array(poly_ls))

"""
Write tiles to files
"""
all_polygons = [Polygon(array_i) for array_i in all_tile_ls]

outlines = gpd.GeoSeries(all_polygons, crs="epsg:2193")
outlines.to_file("tile_outlines.shp")

outlines_wgs = outlines.to_crs(epsg=4326)
outlines_wgs.to_file("tile_outlines.shp")

all_points = [Point(row) for row in all_points_array]
centres = gpd.GeoSeries(all_points, crs="epsg:2193")
centres.to_file("tile_centres.shp")
all_points_z = np.array([point.z for point in all_points])

# Export in alternative format
top_trace_gs = gpd.GeoSeries(top_traces, crs="epsg:2193")
top_trace_wgs = top_trace_gs.to_crs(epsg=4326)

out_alternative_ls = []
for trace, dip, top_depth, bottom_depth in zip(top_trace_wgs.geometry, dips, top_depths, bottom_depths):
    x0, y0 = trace.coords[0][:-1]
    x1, y1 = trace.coords[1][:-1]
    if y1 > y0:
        out_alternative_ls.append([x1, y1, x0, y0, dip, top_depth / -1000, bottom_depth / -1000])
    else:
        out_alternative_ls.append([x0, y0, x1, y1, dip, top_depth / -1000, bottom_depth / -1000])

out_alternative_array = np.array(out_alternative_ls)
index_array = np.array(all_indices[:5000])

#Dataframes provides simpler formatting options
df_indices  = pd.DataFrame(index_array, columns=["along_strike_index", "down_dip_index"])
df_tiles    = pd.DataFrame(out_alternative_array, columns=["lon1(deg)", "lat1(deg)", "lon2(deg)", "lat2(deg)", "dip (deg)", "top_depth (km)", "bottom_depth (km)"])
df_centres  = pd.DataFrame(all_points_array[:5000], columns=["cen_x", "cen_y", "cen_z"])

df_tiles_out = pd.merge(df_indices, df_tiles, left_index=True, right_index=True)
df_centres_out = pd.merge(df_indices, df_centres, left_index=True, right_index=True)

df_tiles_out.to_csv(os.path.join(output_dir, "tile_parameters.csv"), index=False)
df_centres_out.to_csv(os.path.join(output_dir, "tile_centres_nztm.csv"), index=False)

# # TODO set to be installable by pip, to avoid these stupid path strings
# np.savetxt(data_dir + "subduction/tile_parameters.txt", out_array_with_indices, fmt="%.6f",
#            delimiter=" ", header=header_str)
# np.savetxt(data_dir + "subduction/tile_centres_nztm.txt", centres_with_indices, fmt="%.6f",
#            delimiter=" ", header=centre_header)
