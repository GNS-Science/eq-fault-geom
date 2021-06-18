# Something was going wrong with the geotiff reading
from eq_fault_geom.geomio.array_operations import read_gmt_grid
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import os
import pandas as pd

from pyproj import Transformer

transformer = Transformer.from_crs(4326, 2193, always_xy=True)
trans_inv = Transformer.from_crs(2193, 4326, always_xy=True)

# Location of data directory: TODO need to decide whether data are installed with project
data_dir = "../../../../data/"
# data_dir = os.path.expanduser("~/DEV/GNS/eq-fault-geom/data/")
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


def fit_plane_to_points(points: np.ndarray, eps: float=1.0e-5):
    """
    Find best-fit plane through a set of points, after first insuring the plane goes through
    the mean (centroid) of all the points in the array. This is probably better than my
    initial method, since the SVD is only over a 3x3 array (rather than the num_pointsxnum_points
    array).
    Returned values are:
        plane_normal:  Normal vector to plane (A, B, C)
        plane_origin:  Point on plane that may be considered as the plane origin
    """
    # Compute plane origin and subract it from the points array.
    plane_origin = np.mean(points, axis=0)
    x = points - plane_origin

    # Dot product to yield a 3x3 array.
    moment = np.dot(x.T, x)

    # Extract single values from SVD computation to get normal.
    plane_normal = np.linalg.svd(moment)[0][:,-1]
    small = np.where(np.abs(plane_normal) < eps)
    plane_normal[small] = 0.0
    plane_normal /= np.linalg.norm(plane_normal)
    if (plane_normal[-1] < 0.0):
        plane_normal *= -1.0

    return plane_normal, plane_origin


def coupling(lat: float):
    if lat < -32.5:
        return 0.2
    elif lat > -31.:
        return 0.5
    else:
        # Linear gradient in the middle between two uniform areas
        return 0.2 + (0.5 - 0.2) * (lat - -32.5) / (-31. - -32.5)

def convergence(lat: float):
    """
    Linear between 49 mm/yr at -39 to 85 mm/yr at -27.5
    """
    north_lat = -27.5
    south_lat = -39.
    south_conv = 49.
    north_conv = 85.

    return south_conv + (north_conv - south_conv) * (lat - south_lat) / (north_lat - south_lat)

def kermadec_slip_rate(lat: float):
    return convergence(lat) * coupling(lat)


"""
Read and pre process data from files, set parameters for meshing
"""

# Relevant quantities that control tile distribution
# Swath profile half width; for making (slightly) smoothed profiles/cross-sections through interface.
profile_half_width = 5000
# Spacing between down-dip profiles (and therefore tiles) in the along-strike direction
profile_spacing = 10000
# Max distance to select points to fit
search_radius = 1.e4


# Read in grid from subduction interface
tiff = os.path.join(data_dir, "subduction/kermadec_lt50k.grd")
# print(data_dir, tiff, Path(tiff).resolve())
x, y, z = read_gmt_grid(tiff)

# Turn grid into xyz (n x 3) array, by making x and y grids and flattening them.
x_grid, y_grid = np.meshgrid(x, y)
all_xyz_with_nans = np.vstack((x_grid.flatten(), y_grid.flatten(), z.flatten())).T

# Remove points where z is NaN
all_xyz = all_xyz_with_nans[~np.isnan(all_xyz_with_nans).any(axis=1)]

slip_deficit_file = os.path.join(data_dir, "subduction/slip_deficit_hikurangi_1.0.txt")
slip_deficit_df = pd.read_csv(slip_deficit_file, delim_whitespace=True)
slip_deficit_nztm_x, slip_deficit_nztm_y = transformer.transform(slip_deficit_df["#longitude"].to_list(),
                                                                 slip_deficit_df["latitude"].to_list())
slip_deficit_nztm_array = np.vstack((slip_deficit_nztm_x, slip_deficit_nztm_y, slip_deficit_df.depth_km.to_numpy() * 1000.,
                                     slip_deficit_df["slip_deficit_mm/yr"].to_numpy(),
                                     slip_deficit_df["uncertainty_mm/yr"].to_numpy())).T

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
    if across_no_nans.size:
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
        if len(interp_z) > 1:
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
slip_deficit_list = []

kermadec_min_lat = -39.

for centre_point in all_points_array:
    centre_wgs = trans_inv.transform(*centre_point)
    # Find slip deficit
    sd_difference_vectors = slip_deficit_nztm_array[:, :3] - centre_point
    sd_all_values = slip_deficit_nztm_array[:, 3]
    sd_distances = np.linalg.norm(sd_difference_vectors, axis=1)
    nearby_sd = slip_deficit_nztm_array[sd_distances < profile_half_width]
    if nearby_sd.size:
        gaussian_sigma = profile_half_width/3.
        sd_weights = np.exp(-1 * sd_distances[sd_distances < profile_half_width] / (2 * gaussian_sigma))
        sd_values = sd_all_values[sd_distances < profile_half_width]
        sd_average = np.average(sd_values, weights=sd_weights)
    elif centre_wgs[1] > kermadec_min_lat:
        # print(centre_wgs)
        sd_average = kermadec_slip_rate(centre_wgs[1])
    else:
        sd_average = -1000.

    # Find distances of all points from centre
    difference_vectors = all_xyz - centre_point
    distances = np.linalg.norm(difference_vectors, axis=1)
    # Select relevant points
    small_cloud = all_xyz[distances < search_radius]

    # Normal to plane
    normal_i, _ = fit_plane_to_points(small_cloud)

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
    slip_deficit_list.append(sd_average)

    all_tile_ls.append(np.array(poly_ls))

"""
Write tiles to files
"""
all_polygons = [Polygon(array_i) for array_i in all_tile_ls]


outlines = gpd.GeoSeries(all_polygons, crs="epsg:2193")
outlines.to_file("hk_tile_outlines.shp")

outlines_wgs = outlines.to_crs(epsg=4326)
outlines_wgs.to_file("hk_tile_outlines.shp")

all_points = [Point(row) for row in all_points_array]
centres = gpd.GeoDataFrame(slip_deficit_list, geometry=all_points, crs="epsg:2193", columns=["slip_deficit"])

centres_wgs = centres.to_crs(4326)
centres.to_file("hk_tile_centres.shp")
all_points_z = np.array([point.z for point in all_points])

# Export in alternative format
top_trace_gs = gpd.GeoSeries(top_traces, crs="epsg:2193")
top_trace_wgs = top_trace_gs.to_crs(epsg=4326)

out_alternative_ls = []
for trace, dip, top_depth, bottom_depth, slip_deficit_i in zip(top_trace_wgs.geometry, dips, top_depths, bottom_depths,
                                                               slip_deficit_list):
    x0, y0 = trace.coords[0][:-1]
    x1, y1 = trace.coords[1][:-1]
    if y1 > y0:
        out_alternative_ls.append([x1, y1, x0, y0, dip, top_depth / -1000, bottom_depth / -1000, slip_deficit_i])
    else:
        out_alternative_ls.append([x0, y0, x1, y1, dip, top_depth / -1000, bottom_depth / -1000, slip_deficit_i])

out_alternative_array = np.array(out_alternative_ls)
index_array = np.array(all_indices)




# #Dataframes provides simpler formatting options
df_indices  = pd.DataFrame(index_array, columns=["along_strike_index", "down_dip_index"])
df_tiles    = pd.DataFrame(out_alternative_array, columns=["lon1(deg)", "lat1(deg)", "lon2(deg)", "lat2(deg)", "dip (deg)", "top_depth (km)", "bottom_depth (km)", "slip_deficit (mm/yr)"])
df_centres  = pd.DataFrame(all_points_array, columns=["cen_x", "cen_y", "cen_z"])

#extend alt_out with tile geometry
df_tiles_xyz = pd.DataFrame(all_polygons, columns=['tile_geometry'])
df_tiles = pd.merge(df_tiles, df_tiles_xyz, left_index=True, right_index=True)


df_tiles_out = pd.merge(df_indices, df_tiles, left_index=True, right_index=True)
df_centres_out = pd.merge(df_indices, df_centres, left_index=True, right_index=True)

df_tiles_out.to_csv(os.path.join(output_dir, "hk_tile_parameters_10.csv"), index=False)
df_centres_out.to_csv(os.path.join(output_dir, "hk_tile_centres_nztm_10.csv"), index=False)
