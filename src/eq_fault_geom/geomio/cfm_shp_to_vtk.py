from pathlib2 import Path
import string
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import numpy as np
import meshio
# import pdb
# pdb.set_trace()

# Epsilon value.
eps = 5000.0

# Cell type.
cell_type = 'triangle'
# cell_type = 'quad'

# Output directory and file suffix.
output_dir = "../../../data/cfm_shapefile/cfm_vtk"
vtk_suffix = ".vtk"

# Create output directory if it does not exist.
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Which dip and depth values to use.
dip_use = 'Dip_Best'
depth_use = 'Depth_Max'

# Dictionary for replacing numbers with letters.
string_list = string.ascii_uppercase[0:10]
num_replace_dict = {str(idx + 1):letter for idx, letter in enumerate(string_list)}

# Default elevation for bottom of faults.
default_elev = -50000.0

# Default length for horizontal faults.
default_length = 50000.0

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


def calculate_dip_rotation(line: LineString, dip_dir: str):
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


def create_mesh_from_trace(fault_info: pd.Series, line: LineString, dip_rotation: np.ndarray):
    """
    Project along dip vector to get downdip points, then make a mesh using all points.
    """
    # Create dip vector directed East, then rotate to dip direction.
    dip = -np.radians(fault_info[dip_use])
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
    elev_max = -1000.0*fault_info[depth_use]
    if (np.abs(elev_max) < eps):
        elev_max = default_elev
    if (dip != 0.0):
        dist = elev_max/dip_vec[2]
    else:
        dist = default_length

    # Create points at depth.
    pd = pt + dist*dip_vec
    points = np.concatenate((pt, pd), axis=0)

    # Create connectivity.
    if (cell_type == "quad"):
        cells = create_quads(num_trace_points)
    else:
        cells = create_triangles(num_trace_points)

    # Create mesh using meshio.
    mesh = meshio.Mesh(points, cells)

    return mesh


def create_quads(num_trace_points):
    """
    Create quad cells given the number of points at the surface.
    """
    num_cells = num_trace_points - 1
    cell_array = np.zeros((num_cells, 4), dtype=np.int)
    for cell_num in range(num_cells):
        cell_array[cell_num,0] = cell_num
        cell_array[cell_num,1] = cell_num + 1
        cell_array[cell_num,2] = cell_num + num_trace_points + 1
        cell_array[cell_num,3] = cell_num + num_trace_points

    cells = [("quad", cell_array)]

    return cells


def create_triangles(num_trace_points):
    """
    Create quad cells given the number of points at the surface.
    """
    num_quads = num_trace_points - 1
    num_triangles = 2*num_quads
    cell_array = np.zeros((num_triangles, 3), dtype=np.int)
    cell_num = 0
    for quad_num in range(num_quads):
        cell_array[cell_num,0] = quad_num
        cell_array[cell_num,1] = quad_num + 1
        cell_array[cell_num,2] = quad_num + num_trace_points + 1
        cell_num += 1
        cell_array[cell_num,0] = quad_num + num_trace_points + 1
        cell_array[cell_num,1] = quad_num + num_trace_points
        cell_array[cell_num,2] = quad_num
        cell_num += 1

    cells = [("triangle", cell_array)]

    return cells
    
    
def create_stirling_vtk(fault_info: pd.Series, section_id: int, nztm_geometry: LineString):
    """
    Create 3D Stirling fault file from 2D map info.
    """
    # Get dip rotation matrix and create mesh from surface info.
    dip_dir = fault_info["Dip_Dir"]
    dip_rotation = calculate_dip_rotation(nztm_geometry, dip_dir)
    mesh = create_mesh_from_trace(fault_info, nztm_geometry, dip_rotation)

    # Write mesh.
    file_name = fault_info["FZ_Name"].replace(" ", "_")
    # Note this only works for faults numbered 1-9.
    if (file_name[-1].isnumeric()):
        file_list = list(file_name)
        file_list[-1] = num_replace_dict[file_name[-1]]
        file_name = "".join(file_list)
    file_path = Path(file_name)
    output_file = Path.joinpath(output_path, file_path).with_suffix(vtk_suffix)
    meshio.write(output_file, mesh, file_format="vtk", binary=False)

    return

if __name__ == '__main__':
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

