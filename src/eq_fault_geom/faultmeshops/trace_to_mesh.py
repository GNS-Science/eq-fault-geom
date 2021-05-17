from typing import Union

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
import numpy as np

import meshio

from .trace_operations import calculate_dip_rotation


def project_mesh_from_trace(fault_info: pd.Series, cell_type: str = "triangle",
                            cell_fields: Union[str, list, tuple] = None, eps: float = 5000.0,
                            dip_use: str = "Dip_Best", depth_use: str = "Depth_Max",
                            default_elev: float = -50000.0, default_length: float = 50000.0):
    """
    Project along dip vector to get downdip points, then make a mesh using all points.
    """
    # Get dip rotation matrix and create mesh from surface info.
    dip_rotation = calculate_dip_rotation(fault_info)

    # Create dip vector directed East, then rotate to dip direction.
    dip = -np.radians(fault_info[dip_use])
    dip_vec_east = np.array([np.cos(dip), 0.0, np.sin(dip)], dtype=np.float64)
    dip_vec = np.dot(dip_rotation, dip_vec_east)
    dip_vec = dip_vec/np.linalg.norm(dip_vec)  # Normalize for good luck.

    # Get surface trace coordinates. We assume that z=0 at surface.
    (xl, yl) = fault_info.geometry.xy
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

    num_cells = cells[0][1].shape[0]

    # Create cell data, if requested.
    cell_data = create_cell_data(fault_info, num_cells, cell_fields)

    # Create mesh using meshio.
    mesh = meshio.Mesh(points, cells, cell_data=cell_data)

    return mesh


def create_cell_data(fault_info: pd.Series, num_cells: int, cell_fields: Union[str, list, tuple] = None):
    """
    Create cell fields from fault metadata, if requested.
    """
    cell_data_dict = None
    if (cell_fields == None):
        return cell_data_dict

    if (type(cell_fields) == 'str'):
        use_fields = [cell_fields]
    else:
        use_fields = [i for i in cell_fields]

    num_fields = len(use_fields)
    cell_data_dict = {}
    
    for field in use_fields:
        val = fault_info[field]
        val_type = type(val)
        val_array = val*np.ones(num_cells, dtype=val_type)
        cell_data_dict[field] = [val_array]
    
    return cell_data_dict


def create_quads(num_trace_points: int):
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


def create_triangles(num_trace_points: int):
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
    
