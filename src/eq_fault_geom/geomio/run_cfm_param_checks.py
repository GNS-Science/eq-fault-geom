import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from eq_fault_geom.geomio.cfm_faults import CfmMultiFault

shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/cfm_linework/NZ_CFM_v0_3_170620.shp"

# Polygons to exclude faults from XML
exclude_shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/test_bop.shp"
exclude_df = gpd.GeoDataFrame.from_file(exclude_shp)
poly_ls = list(exclude_df.geometry)

# # To read in Matt's polygon

matt_array = np.array([[-36.17, 177.25],
                       [-36.17, 178.14],
                       [-37.53, 177.31],
                       [-39.78, 175.38],
                       [-39.78, 174.97],
                       [-39.22, 175.29],
                       [-36.17, 177.25]])

matt_poly = [Polygon([(row[1], row[0]) for row in matt_array])]
matt_gs = gpd.GeoSeries(matt_poly[0], crs=4326)
matt_gs.to_file("matt_poly.shp")

data = CfmMultiFault.from_shp(shp, exclude_regions=poly_ls)
xml = data.to_opensha_xml(exclude_subduction=True)
with open("test3.xml", "wb") as f:
    f.write(xml)
