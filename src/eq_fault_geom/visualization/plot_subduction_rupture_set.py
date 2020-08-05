import geopandas as gpd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import mpl_toolkits.mplot3d.axes3d as p3

from matplotlib import cm
import os
import sys 

from eq_fault_geom.geomops.rupture_set.fault_section import SheetFault, FaultSubSection, FaultSubSectionFactory

# ref 1) https://nickcharlton.net/posts/drawing-animating-shapes-matplotlib.html
# ref 2) https://matplotlib.org/gallery/animation/simple_3danim.html

from matplotlib import animation
# matplotlib.use('TkAgg')

# Location of data directory: TODO need to decide whether data are installed with project
# data_dir =  os.getcwd()
data_dir = os.path.expanduser("~/DEV/GNS/opensha/nshm-nz-opensha/data/subduction")

# Shapefile containing coordinates of tile outlines (in lat lon)
outline_file = os.path.join(data_dir, "tile_outlines.zip") 

# Read in data
tile_outlines = gpd.GeoDataFrame.from_file("zip://" + outline_file + "!tile_outlines.shp")
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
patch_colour = ""
# Patch transparency (between 0 and 1)
patch_alpha = 0.5
# Outline colour and width
line_colour = "gray"
line_width = 0.2

#rupture patches
rupture_alpha = 0
rupture_colour = "red"

###############################
#
#
tile_params = os.path.join(data_dir, "tile_parameters.csv")
factory = FaultSubSectionFactory()
sf = SheetFault("Hikurangi")\
		.build_surface_from_csv(factory, open(tile_params))

# Variations on 'nearly rectangular' rupture sets ..
#
# scale: determines ratio of rectangle to original tile size (10km*10km).
# aspect: ratio of 'along-strike' tiles to 'down-dip' tiles per rectangle.
# interval: how many tiles to advance both col-wise and row-wise
# min_fill_factor: how many tiles are needed to make up a valid 'rectangle'
#
specs = [dict(scale=2, aspect=2, interval=2, min_fill_factor=0.7),
 	     dict(scale=4, aspect=2, interval=3, min_fill_factor=0.7),
	     dict(scale=4, aspect=4, interval=3, min_fill_factor=0.7),
	     dict(scale=8, aspect=2, interval=4, min_fill_factor=0.55),
	     dict(scale=16, aspect=1, interval=4, min_fill_factor=0.33),
	     dict(scale=16, aspect=2, interval=4, min_fill_factor=0.33)]

#build the rupture set
ruptures = []
for spec in specs:
	ruptures.extend([r for r in sf.get_rupture_ids(spec)])
#
#
###############################

# Create collections for matplotlib
# patch_collection = Poly3DCollection(all_tile_ls, alpha=patch_alpha, facecolors=patch_colour)
line_collection = Line3DCollection(all_tile_ls, linewidths=line_width, colors=line_colour)
patch_collection = Poly3DCollection(all_tile_ls, alpha=patch_alpha, facecolors=patch_colour)
rupture_collection = Poly3DCollection([], alpha=rupture_alpha, facecolors=rupture_colour)


# # Create colormap by z (limits set by max and min patch centre depths)
# colormap = cm.ScalarMappable(cmap=cm.magma)
# colormap.set_array(np.array([min(tile_centre_z), max(tile_centre_z)]))
# colormap.set_clim(vmin=min(tile_centre_z), vmax=max(tile_centre_z))
# patch_collection.set_facecolor(colormap.to_rgba(tile_centre_z, alpha=patch_alpha))


# Plot data
plt.close("all")
# Create figure and axis objects
fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

ax = p3.Axes3D(fig)

ax.add_collection3d(line_collection)
ax.add_collection3d(patch_collection)
ax.add_collection3d(rupture_collection)

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


def animate(i, rupture_collection):
	rupture_tiles = [sf.sub_sections[ssid].tile_coords for ssid in ruptures[i]]
	rupture_collection.set_verts(rupture_tiles)
	return rupture_collection,

# Factor by which to vertically exaggerate z axis of plot
vertical_exaggeration = 2

# Set x, y, z limits so plot has equal aspect ratio
ax.set_ylim((y1, y2))
ax.set_xlim((x1, x2))
ax.set_zlim((-(1/vertical_exaggeration) * plot_width, 10))

ax.view_init(elev=30., azim=80)

#init_func=init, 
anim = animation.FuncAnimation(fig, animate,
							   fargs=(rupture_collection,),
                               frames=len(ruptures), 
                               interval=10,
                               blit=False)

# Show figure
plt.show()
#anim.save('/tmp/approx_rectangular_ruptures.mp4', fps=15)

