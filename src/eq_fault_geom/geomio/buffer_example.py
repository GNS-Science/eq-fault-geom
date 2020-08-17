import geopandas as gpd

# Example file; should work on whole dataset too
shp_file = "/Users/arh79/PycharmProjects/eq-fault-geom/data/cfm_shapefile/cfm_lower_n_island.shp"

# read in data
shp_df = gpd.GeoDataFrame.from_file(shp_file)
# Sort alphabetically by name
sorted_df = shp_df.sort_values("Name")
# Reset index to line up with alphabetical sorting
sorted_df = sorted_df.reset_index(drop=True)
# Reproject traces into lon lat
sorted_wgs = sorted_df.to_crs(epsg=4326)

geometry_nztm = sorted_df.geometry
# Create 1 km buffer around each fault
buffered = geometry_nztm.buffer(1000)
# Convert NZTM to lon lat
buffered_wgs = buffered.to_crs(epsg=4326)

# Turn into dictionary array
polygon_outlines = [poly.exterior for poly in buffered_wgs]
polygon_lonlat = {i:[poly.xy] for i, poly in enumerate(polygon_outlines)}