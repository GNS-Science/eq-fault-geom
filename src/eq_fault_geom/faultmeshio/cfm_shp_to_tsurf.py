from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import meshio
from eq_fault_geom.faultmeshops.trace_to_mesh import project_mesh_from_trace
import eq_fault_geom.faultmeshio.tsurf as ts
# import pdb
# pdb.set_trace()


# Output directory and file suffix.
output_dir = "../../../data/cfm_shapefile/cfm_tsurf"
tsurf_suffix = ".ts"

# Cell type.
cell_type = 'triangle'

# Create output directory if it does not exist.
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

# Example file; should work on whole dataset too
shp_file = "../../../data/cfm_shapefile/cfm_lower_n_island.shp"
    
# read in data
shp_df = gpd.GeoDataFrame.from_file(shp_file)

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
    output_file = Path.joinpath(output_path, file_path).with_suffix(tsurf_suffix)
    tsurf_mesh.write(output_file)

