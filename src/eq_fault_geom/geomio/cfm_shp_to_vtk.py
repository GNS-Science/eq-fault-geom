from pathlib2 import Path
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import meshio
# import pdb
# pdb.set_trace()

# Epsilon value.
eps = 5000.0

# Output directory.
output_dir = "../../../data/cfm_shapefile/cfm_vtk"
vtk_suffix = ".vtk"

# Create output directory if it does not exist.
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

def calculate_dip_rotation(line: LineString):
    """
    Calculate slope of fault trace in NZTM, then add 90 to get dip direction.
    Form 3D rotation matrix from dip direction.
    :param line: Linestring object
    :return:
    """
    # Get coordinates
    x, y = line.xy

    # Calculate gradient of line in 2D
    p = np.polyfit(x, y, 1)
    gradient = p[0]

    # Gradient to bearing
    bearing = np.pi - np.arctan2(gradient, 1)

    if x[0] > x[-1]:
        bearing += np.pi

    # Rotation matrix.
    cosb = np.cos(bearing)
    sinb = np.sin(bearing)
    rot_mat = np.array([[cosb, sinb, 0.0],
                        [-sinb, cosb, 0.0],
                        [0.0, 0.0, 1.0]], dtype=np.float64)

    return rot_mat


def create_mesh_from_trace(fault_info: pd.Series, line: LineString, dip_rotation: np.ndarray):
    """
    Project along dip vector to get downdip points, then make a mesh using all points.
    """
    # Create dip vector directed East, then rotate to dip direction.
    dip = -np.radians(fault_info["Dip_Best"])
    dip_vec_east = np.array([np.cos(dip), 0.0, np.sin(dip)], dtype=np.float64)
    dip_vec = np.dot(dip_rotation, dip_vec_east)
    dip_vec = dip_vec/np.linalg.norm(dip_vec)  # Normalize for good luck.

    # Get surface trace coordinates. We assume that z=0 at surface.
    (xl, yl) = line.xy
    xt = np.array(xl)
    yt = np.array(yl)
    num_trace_points = xt.shape[0]
    num_points = 2*num_trace_points
    pt = np.column_stack((xt, yt, np.zeros_like(xt)))

    # Get distance along dip vector.
    elev_max = -1000.0*fault_info["Depth_Max"]
    if (np.abs(elev_max) < eps):
        elev_max = -50000.0
    dist = elev_max/dip_vec[2]

    # Create points at depth.
    pd = pt + dist*dip_vec
    points = np.concatenate((pt, pd), axis=0)

    # Create connectivity.
    num_cells = num_trace_points - 1
    cellArray = np.zeros((num_cells, 4), dtype=np.int)
    for cell_num in range(num_cells):
        cellArray[cell_num,0] = cell_num
        cellArray[cell_num,1] = cell_num + 1
        cellArray[cell_num,2] = cell_num + num_trace_points + 1
        cellArray[cell_num,3] = cell_num + num_trace_points

    # Create meshio mesh.
    cells = [("quad", cellArray)]
    mesh = meshio.Mesh(points, cells)

    return mesh
    
    
def create_stirling_vtk(fault_info: pd.Series, section_id: int, nztm_geometry: LineString):
    """
    Create 3D Stirling fault file from 2D map info.
    """
    # Get dip rotation matrix and create mesh from surface info.
    dip_rotation = calculate_dip_rotation(nztm_geometry)
    mesh = create_mesh_from_trace(fault_info, nztm_geometry, dip_rotation)

    # Write mesh.
    file_name = fault_info["Name"].replace(" ", "_")
    file_path = Path(file_name)
    output_file = Path.joinpath(output_path, file_path).with_suffix(vtk_suffix)
    meshio.write(output_file, mesh, file_format="vtk", binary=False)

    return

# Example file; should work on whole dataset too
shp_file = "../../../data/cfm_shapefile/cfm_lower_n_island.shp"

# read in data
shp_df = gpd.GeoDataFrame.from_file(shp_file)

# Sort alphabetically by name
sorted_df = shp_df.sort_values("Name")

# Reset index to line up with alphabetical sorting
sorted_df = sorted_df.reset_index(drop=True)

# Reproject traces into lon lat
sorted_wgs = sorted_df.to_crs(epsg=4326)

# Loop through faults, creating a VTK file for each.
for i, fault in sorted_wgs.iterrows():
    # Extract NZTM line for dip direction calculation/could be done in a better way, I'm sure
    nztm_geometry_i = sorted_df.iloc[i].geometry

    # Create Stirling fault segment and write VTK file.
    create_stirling_vtk(fault, section_id=i, nztm_geometry=nztm_geometry_i)

