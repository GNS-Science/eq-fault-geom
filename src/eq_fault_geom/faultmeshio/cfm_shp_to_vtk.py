from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np
import meshio
from eq_fault_geom.faultmeshops.trace_to_mesh import project_mesh_from_trace
# import pdb
# pdb.set_trace()


# Output directory and file suffix.
output_dir = "../../../data/cfm_shapefile/cfm_vtk"
vtk_suffix = ".vtk"

# Cell fields (fault metadata) to include with mesh.
cell_fields = ['Depth_Best', 'Depth_Max', 'Depth_Min', 'Dip_Best', 'Dip_Max', 'Dip_Min', 'Number',
               'Qual_Code', 'Rake_Best', 'Rake_Max', 'Rake_Min', 'SR_Best', 'SR_Max', 'SR_Min']

# Cell type.
cell_type = 'triangle'
# cell_type = 'quad'

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

# Loop through faults, creating a VTK file for each.
for i, fault in sorted_df.iterrows():
    # Create fault segment by projecting mesh downdip.
    faultmesh = project_mesh_from_trace(fault, cell_type=cell_type, cell_fields=cell_fields)

    # Write mesh.
    file_name = fault["FZ_Name"].replace(" ", "_")

    file_path = Path(file_name)
    output_file = Path.joinpath(output_path, file_path).with_suffix(vtk_suffix)
    meshio.write(output_file, faultmesh, file_format="vtk", binary=False)

