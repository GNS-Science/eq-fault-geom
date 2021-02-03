#!/usr/bin/env python3
"""
Simple script to convert entries from the current CFM shapefile into
multiple tsurf files.
"""
from pathlib import Path
import argparse
import geopandas as gpd
import pandas as pd
import numpy as np

import meshio

from eq_fault_geom.faultmeshops.trace_to_mesh import project_mesh_from_trace

# import pdb
# pdb.set_trace()

# File suffix.
vtk_suffix = ".vtk"

# ----------------------------------------------------------------------

def shp_to_multi_vtk(in_path, out_path, cell_type, output_fields):
    """
    Read CFM shapefile and create multiple VTK files in specified directory.
    """
    
    # read in data
    shp_df = gpd.GeoDataFrame.from_file(in_path)

    # Sort alphabetically by name
    try:
        sorted_df = shp_df.sort_values("Name")
    except:
        sorted_df = shp_df.sort_values("FZ_Name")

    # Reset index to line up with alphabetical sorting
    sorted_df = sorted_df.reset_index(drop=True)

    # Loop through faults, creating a VTK file for each.
    for i, fault in sorted_df.iterrows():
        # Create fault segment by projecting mesh downdip.
        faultmesh = project_mesh_from_trace(fault, cell_type=cell_type, cell_fields=output_fields)

        # Write mesh.
        file_name = fault["FZ_Name"].replace(" ", "_")

        file_path = Path(file_name)
        output_file = Path.joinpath(out_path, file_path).with_suffix(vtk_suffix)
        meshio.write(output_file, faultmesh, file_format="vtk", binary=False)

    return

# ======================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert files from the CFM shapefile to VTK format.')
    parser.add_argument("-i", "--in_file", action="store", 
                        dest="in_file", required=True, help="input shapefile")
    parser.add_argument("-o", "--out_dir", action="store", 
                        dest="out_dir", required=True, help="output directory")
    parser.add_argument("-c", "--cell_type", required=False, help="output cell type (triangle or quad)",
                        default="triangle", action="store", dest="cell_type", choices=['triangle', 'quad'])
    parser.add_argument("-f", "--cell_fields", required=False, help="output cell fields",
                        default="all_numerical", action="store", dest="cell_fields")

    args = parser.parse_args()

    # Input file and output directory.
    in_path = Path(args.in_file)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Cell info.
    cell_type = args.cell_type
    cell_fields = args.cell_fields

    # Cell fields (fault metadata) to include with mesh.
    output_fields = []
    if (cell_fields == "all_numerical"):
        output_fields = ['Depth_Best', 'Depth_Max', 'Depth_Min', 'Dip_Best', 'Dip_Max', 'Dip_Min', 'Number',
                         'Qual_Code', 'Rake_Best', 'Rake_Max', 'Rake_Min', 'SR_Best', 'SR_Max', 'SR_Min']
    else:
        output_fields = [field for field in cell_fields.split()]

    # Call function to generate VTKs from shapefile.
    shp_to_multi_vtk(in_path, out_path, cell_type, output_fields)
    
