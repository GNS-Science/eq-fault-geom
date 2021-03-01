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
import eq_fault_geom.faultmeshio.tsurf as ts

# import pdb
# pdb.set_trace()

# File suffix.
tsurf_suffix = ".ts"

# Cell type.
cell_type = 'triangle'

# ----------------------------------------------------------------------
def shp_to_multi_tsurf(in_path, out_path):
    """
    Read CFM shapefile and create multiple Tsurf files in specified directory.
    """

    # read in data
    shp_df = gpd.GeoDataFrame.from_file(in_path)

    # Sort alphabetically by name
    sorted_df = shp_df.sort_values("Name")

    # Reset index to line up with alphabetical sorting
    sorted_df = sorted_df.reset_index(drop=True)

    # Loop through faults, creating a Tsurf file for each.
    for i, fault in sorted_df.iterrows():
        # Create fault segment by projecting mesh downdip.
        faultmesh = project_mesh_from_trace(fault, cell_type=cell_type, cell_fields=None)

        # Get fault name.
        fault_name = fault["FZ_Name"].replace(" ", "_")

        # Create Tsurf mesh.
        x = faultmesh.points[:,0]
        y = faultmesh.points[:,1]
        z = faultmesh.points[:,2]
        cells = faultmesh.cells
        tsurf_mesh = ts(x, y, z, cells, name=fault_name)

        # Write Tsurf mesh.
        file_path = Path(fault_name)
        output_file = Path.joinpath(out_path, file_path).with_suffix(tsurf_suffix)
        tsurf_mesh.write(output_file)

    return

# ======================================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Convert files from the CFM shapefile to Tsurf format.')
    parser.add_argument("-i", "--in_file", action="store", 
                        dest="in_file", required=True, help="input shapefile")
    parser.add_argument("-o", "--out_dir", action="store", 
                        dest="out_dir", required=True, help="output directory")

    args = parser.parse_args()

    in_path = Path(args.in_file)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Call function to generate Tsurfs from shapefile.
    shp_to_multi_tsurf(in_path, out_path)
