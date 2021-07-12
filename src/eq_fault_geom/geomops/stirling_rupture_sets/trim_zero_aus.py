"""
To trim zero-slip-rate faults from CFM
"""
import geopandas as gpd

cfm = gpd.read_file("../../../../data/cfm_shapefile/cfm_0_9.gpkg")
nonzero_sr = cfm[["SR_pref", "SR_min", "SR_max"]] > 0
cfm_nonzero = cfm[nonzero_sr.any(axis=1)]

cfm_no_aus = cfm_nonzero[cfm_nonzero.Fault_stat != "A-US"]
