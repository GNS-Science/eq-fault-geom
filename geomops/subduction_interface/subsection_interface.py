from icp_error.io.array_operations import read_tiff
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm


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


# Read in grid from subduction interface
x, y, z = read_tiff("williams_0_005_nztm.tif")
# Multiply z coordinates by 1000 so that everything is in metres
z *= 1000

# Turn grid into xyz (n x 3) array, by making x and y grids and flattening them.
x_grid, y_grid = np.meshgrid(x, y)
all_xyz_with_nans = np.vstack((x_grid.flatten(), y_grid.flatten(), z.flatten())).T

# Remove points where z is NaN
all_xyz = all_xyz_with_nans[~np.isnan(all_xyz_with_nans).any(axis=1)]

# Read shapefile: line that gives overall strike of subduction zone
# Included so that easy to fiddle with in GIS
overall_trace = gpd.GeoDataFrame.from_file("overall_trace.shp")
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

# Calculate distances of coordinates in mesh from corner point
along_dists = (x_grid - corner[0]) * along_overall[0] + (y_grid - corner[1]) * along_overall[1]
across_dists = (x_grid - corner[0]) * across_vec[0] + (y_grid - corner[1]) * across_vec[1]

# Swath profile half width; for making (slightly) smoothed profiles/cross-sections through interface.
profile_half_width = 2000
# Spacing between down-dip profiles (and therefore tiles) in the along-strike direction
profile_spacing = 10000

# Find start and end locations along strike (when z values stop being NaNs and start again)
start_along = min(along_dists[~np.isnan(z)])
end_along = max(along_dists[~np.isnan(z)])

# Space profiles evenly along strike
along_spaced = np.arange(start_along + profile_spacing/2, end_along, profile_spacing)

all_points_ls = []

for along in along_spaced:
    row_end = corner + along * along_overall

    along_min = along - profile_half_width
    along_max = along + profile_half_width

    in_swath = np.logical_and(along_dists >= along_min, along_dists <= along_max)
    swath_across = across_dists[in_swath]
    swath_z = z[in_swath]

    across_no_nans = swath_across[~np.isnan(swath_z)]

    start_across = min(across_no_nans)
    end_across = max(across_no_nans)

    initial_spacing = np.arange(start_across, end_across, profile_half_width)

    across_vs_z = np.vstack((across_no_nans, swath_z[~np.isnan(swath_z)])).T
    sorted_coords = across_vs_z[across_vs_z[:, 0].argsort()]

    interp_z = np.interp(initial_spacing, sorted_coords[:, 0], sorted_coords[:, 1])

    interp_line = LineString(np.vstack((initial_spacing, interp_z)).T)

    interpolation_distances = np.arange(profile_spacing/2, interp_line.length, profile_spacing)
    interpolated_points = [interp_line.interpolate(distance) for distance in interpolation_distances]

    interpolated_x = np.array([point.x for point in interpolated_points])
    interpolated_z_values = np.array([point.y for point in interpolated_points])

    point_xys = np.array([row_end + across_i * across_vec for across_i in interpolated_x])
    point_xyz = np.vstack((point_xys.T, interpolated_z_values)).T

    all_points_ls.append(point_xyz)

all_points_array = np.vstack(all_points_ls)

search_radius = 1e4

all_tile_ls = []

for centre_point in all_points_array:
    difference_vectors = all_xyz - centre_point
    distances = np.linalg.norm(difference_vectors, axis=1)
    small_cloud = all_xyz[distances < search_radius]

    u_norm_i = fit_plane_svd(small_cloud)

    normal_i = u_norm_i
    if normal_i[-1] < 0:
        normal_i *= -1

    strike_vector = np.cross(normal_i, np.array([0, 0, -1]))
    strike_vector[-1] = 0
    strike_vector /= np.linalg.norm(strike_vector)

    down_dip_vector = np.cross(normal_i, strike_vector)
    if down_dip_vector[-1] > 0:
        down_dip_vector *= -1

    dip = np.degrees(np.arctan(-1 * down_dip_vector[-1] / np.linalg.norm(down_dip_vector[:-1])))

    poly_ls = []
    for i, j in zip([1, 1, -1, -1], [1, -1, -1, 1]):
        corner_i = centre_point + (i * strike_vector + j * down_dip_vector) * profile_spacing / 2
        poly_ls.append(corner_i)

    all_tile_ls.append(np.array(poly_ls))

all_polygons = [Polygon(array_i) for array_i in all_tile_ls]

outlines = gpd.GeoSeries(all_polygons, crs="epsg:2193")
outlines.to_file("tile_outlines.shp")

outlines_wgs = outlines.to_crs(epsg=4326)
outlines_wgs.to_file("tile_outlines.shp")

all_points = [Point(row) for row in all_points_array]
centres = gpd.GeoSeries(all_points, crs="epsg:2193")
centres.to_file("tile_centres.shp")
all_points_z = np.array([point.z for point in all_points])