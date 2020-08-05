import geopandas as gpd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import cm
import os

# matplotlib.use('TkAgg')

# Location of data directory: TODO need to decide whether data are installed with project
# data_dir = "/Users/arh79/PycharmProjects/eq-fault-geom/data/"
data_dir =  os.getcwd()

# Shapefile containing coordinates of tile outlines (in lat lon)
outline_file = os.path.join(data_dir, "tile_outlines.shp")

# Read in data
tile_outlines = gpd.GeoDataFrame.from_file(outline_file)
# Convert to NZTM
tile_outlines_nztm = tile_outlines.to_crs(epsg=2193)

# Extract coordinates and place in list that can be read by matplotlib
all_tile_ls = [list(tile.boundary.coords) for tile in tile_outlines_nztm.geometry]

# Coordinates of tile corners
all_coordinates = np.array([tile[:-1] for tile in all_tile_ls])
# Coordinates of tile centres
tile_centres = np.mean(all_coordinates, axis=1)
# Z values of centres (for colormap)
tile_centre_z = tile_centres[:, -1]

# Dummy patch colour, removed later
patch_colour = "gray"
# Patch transparency (between 0 and 1)
patch_alpha = 0.5
# Outline colour and width
line_colour = "k"
line_width = 0.2


# Create collections for matplotlib
patch_collection = Poly3DCollection(all_tile_ls, alpha=patch_alpha, facecolors=patch_colour)
line_collection = Line3DCollection(all_tile_ls, linewidths=line_width, colors=line_colour)

# Create colormap by z (limits set by max and min patch centre depths)
# colormap = cm.ScalarMappable(cmap=cm.magma)
# colormap.set_array(np.array([min(tile_centre_z), max(tile_centre_z)]))
# colormap.set_clim(vmin=min(tile_centre_z), vmax=max(tile_centre_z))
# patch_collection.set_facecolor(colormap.to_rgba(tile_centre_z, alpha=patch_alpha))

# Plot data
plt.close("all")
# Create figure and axis objects
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Find bottom left corner of z=0 plane
bounds = tile_outlines_nztm.bounds
x1, y1 = min(bounds.minx), min(bounds.miny)
x_range = max(bounds.maxx) - x1
y_range = max(bounds.maxy) - y1

# Width is the larger of the x and y ranges
plot_width = max([x_range, y_range])
# Top right corner of z=0 plane
x2 = x1 + plot_width
y2 = y1 + plot_width

# Add data to plot
ax.add_collection3d(patch_collection)
ax.add_collection3d(line_collection)


# Factor by which to vertically exaggerate z axis of plot
vertical_exaggeration = 5

# Set x, y, z limits so plot has equal aspect ratio
ax.set_ylim((y1, y2))
ax.set_xlim((x1, x2))
ax.set_zlim((-(1/vertical_exaggeration) * plot_width, 10))

# Show figure
plt.show()
