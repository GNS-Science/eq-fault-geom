from eq_fault_geom.geomio.cfm_faults import CfmMultiFault
import geopandas as gpd

shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/cfm_linework/NZ_CFM_v0_3_170620.shp"

# Polygons to exclude faults from XML
exclude_shp = "/Users/arh79/PycharmProjects/eq-fault-geom/src/eq_fault_geom/geomio/bop_mask_wgs.shp"
exclude_df = gpd.GeoDataFrame.from_file(exclude_shp)
poly_ls = list(exclude_df.geometry[0])

data = CfmMultiFault.from_shp(shp, exclude_regions=poly_ls)
xml = data.to_opensha_xml(exclude_subduction=True)
with open("test3.xml", "wb") as f:
    f.write(xml)
