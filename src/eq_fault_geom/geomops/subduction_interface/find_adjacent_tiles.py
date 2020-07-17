import numpy as np
import pickle

# Location of data directory: TODO need to decide whether data are installed with project
data_dir = "/Users/arh79/PycharmProjects/eq-fault-geom/data/"

# Overall strike of subduction interface
overall_trend = 40.90205712924798 + 180

tile_centres = np.genfromtxt(data_dir + "subduction/tile_centres_nztm.txt")
out_file = data_dir + "subduction/adjacent_tiles.pkl"
# along_i, down_i, cen_x, cen_y, cen_z = [tile_centres[:, i] for i in range(5)]

# We'll need to experiment with this value, should be less than 2e4 for single-tile radius
search_radius = 1.4e4

# To store indices of nearby patches
nearby_dic = {}

# Loop through tile centres, finding nearby patches
for centre in tile_centres:
    # Coordinates of tile centre
    cen_xyz = centre[2:]
    # Relative positions of all other tile centres
    difference_vectors = tile_centres[:, 2:] - cen_xyz

    # Distances to other tile centres
    distances = np.linalg.norm(difference_vectors, axis=1)
    # Select only tile centres below search radius from tile of interest
    # Minimum distance of 10 m to avoid selecting the tile centre (cannot be adjacent to itself)
    nearby_indices = np.where(np.logical_and(distances < search_radius, distances > 10.))[0]
    nearby = tile_centres[nearby_indices, :]

    # Relative positions of nearby tiles
    nearby_differences = difference_vectors[nearby_indices, :]
    # Azimuths of vectors to nearby tiles
    angles = np.degrees(np.arctan2(nearby_differences[:, 1], nearby_differences[:, 0]))
    angles[angles < 0] += 360

    # Sort adjacent tiles (currently anticlockwise from due east, I think)
    sorted_nearby = nearby[angles.argsort(), :]

    # Get indices of adjacent tiles
    nearby_along_down_indices = [(int(along), int(down)) for along, down in zip(sorted_nearby[:, 0],
                                                                                sorted_nearby[:, 1])]
    centre_indices = (int(centre[0]), int(centre[1]))

    # Add to dictionary, with index of "centre" tile as key
    nearby_dic[centre_indices] = nearby_along_down_indices

# Write dictionary to file using pickle
pickle.dump(nearby_dic, open(out_file, "wb"))





