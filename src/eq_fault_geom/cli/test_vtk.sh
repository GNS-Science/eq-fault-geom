#!/bin/bash
python ./cfm_shp_to_vtk.py --in_file=../../../data/cfm_shapefile/cfm_lower_n_island.shp --out_dir=out1_vtk
python ./cfm_shp_to_vtk.py --in_file=../../../data/cfm_shapefile/cfm_lower_n_island.shp --out_dir=out2_vtk --cell_type=quad --cell_fields='Depth_Best Dip_Best Number Rake_Best SR_Best'
