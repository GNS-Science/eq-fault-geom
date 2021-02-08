import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

from eq_fault_geom.geomio.cfm_faults import CfmMultiFault

# shp = ("/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/CFM_v0_5_Review/"
       # "Shapefiles/NZ_CFM_v0_5_290121.shp")
shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/cfm_linework/NZ_CFM_v0_3_170620.shp"

# Polygons to exclude faults from XML
exclude_shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/test_bop.shp"
exclude_df = gpd.GeoDataFrame.from_file(exclude_shp)
# Change to WGS for use with Matt's TVZ polygon
exclude_df_wgs = exclude_df.to_crs(epsg=4326)
poly_ls = list(exclude_df_wgs.geometry)

# To read in Matt's TVZ polygon
matt_array = np.array([[-36.17, 177.25],
                       [-36.17, 178.14],
                       [-37.53, 177.31],
                       [-39.78, 175.38],
                       [-39.78, 174.97],
                       [-39.22, 175.29],
                       [-36.17, 177.25]])

# Polygon requires lon lat (rather than lat lon)
matt_poly = Polygon([(row[1], row[0]) for row in matt_array])
poly_ls.append(matt_poly)

# # Write to shapefile for visualization in GIS
# matt_gs = gpd.GeoSeries(matt_poly[0], crs=4326)
# matt_gs.to_file("matt_poly.shp")

# Width of trace buffer (in metres)
buffer_width = 5000.

# Read and write data
data = CfmMultiFault.from_shp(shp, exclude_regions=poly_ls)
polygons = [fault.combined_buffer_polygon(buffer_width) for fault in data.faults if abs(fault.down_dip_vector[-1]) > 1.e-3]
polygons_gdf = gpd.GeoDataFrame(geometry=polygons, crs=4326)
polygons_gdf.to_file("fault_buffers.shp")


xml_buffer = data.to_opensha_xml(exclude_subduction=True, buffer_width=buffer_width, write_buffers=True)
with open("cfm_0_3_buffer.xml", "wb") as f:
    f.write(xml_buffer)

xml_nobuffer = data.to_opensha_xml(exclude_subduction=True, buffer_width=buffer_width, write_buffers=False)
with open("cfm_0_3_no_buffer.xml", "wb") as f:
    f.write(xml_nobuffer)