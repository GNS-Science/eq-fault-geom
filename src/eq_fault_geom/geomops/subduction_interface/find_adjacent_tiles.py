import numpy as np
import pickle
import os
import pandas as pd

# Location of data directory: TODO need to decide whether data are installed with project
# data_dir = "/Users/arh79/PycharmProjects/eq-fault-geom/data/"
data_dir = os.getcwd() #os.path.expanduser("~/DEV/GNS/eq-fault-geom/data/")
output_dir = os.getcwd()


# Overall strike of subduction interface
overall_trend = 40.90205712924798 + 180

# tile_centres = np.genfromtxt(data_dir + "subduction/tile_centres_nztm.txt")
df_tile_centres = pd.read_csv(os.path.join(data_dir, "tile_centres_nztm.csv"), header=0)
df_tile_params =  pd.read_csv(os.path.join(data_dir, "tile_parameters.csv"), header=0)
# out_file = data_dir + "subduction/adjacent_tiles.pkl"
# along_i, down_i, cen_x, cen_y, cen_z = [tile_centres[:, i] for i in range(5)]

# # We'll need to experiment with this value, should be less than 2e4 for single-tile radius
#search_radius = 2.5e4

# To store indices of nearby patches
nearby_dic = {}
nearby_list = []

# Loop through tile centres, finding nearby patches
for centre in df_tile_centres.values:
    centre_indices = (int(centre[0]), int(centre[1]))
    # Coordinates of tile centre
    cen_xyz = centre[2:]
    # Relative positions of all other tile centres
    difference_vectors = df_tile_centres.values[:, 2:] - cen_xyz
    # Distances to other tile centres
    # distances = np.linalg.norm(difference_vectors, axis=1)
    # Select only tile centres below search radius from tile of interest
    # Minimum distance of 10 m to avoid selecting the tile centre (cannot be adjacent to itself)
    #nearby_indices = np.where(np.logical_and(distances < search_radius, distances > 1.))[0]
    #nearby = df_tile_centres.values[nearby_indices, :]
    # Relative indicesall other tiles
    dip_diffs = df_tile_centres.values[:, 0:1] - centre_indices[0]
    strike_diffs = df_tile_centres.values[:, 1:2] - centre_indices[1]
    adjacent_tiles   = np.logical_and(
                        abs(dip_diffs) <=1, 
                        abs(strike_diffs) <=1)
    # adjacent_tiles   = abs(dip_diffs) <=1
    nearby_indices = np.where(adjacent_tiles)[0]
    nearby = df_tile_centres.values[nearby_indices, :]
    # Relative positions of nearby tiles
    nearby_differences = difference_vectors[nearby_indices, :]
    # Azimuths of vectors to nearby tiles
    angles = np.degrees(np.arctan2(nearby_differences[:, 0], nearby_differences[:, 1]))
    angles[angles < 0] += 360
    # Sort adjacent tiles (currently anticlockwise from due east, I think)
    sorted_nearby = nearby[angles.argsort(), :]
    # Get indices of adjacent tiles
    nearby_along_down_indices = [(int(along), int(down)) for along, down in zip(sorted_nearby[:, 0],
                                                                             sorted_nearby[:, 1])]
    nearby_along_down_indices.pop (nearby_along_down_indices.index(centre_indices))
    # Add to dictionary, with index of "centre" tile as key
    #nearby_dic[centre_indices] = nearby_along_down_indices
    try:
        assert(len(nearby_along_down_indices) <=8)
        assert(len(nearby_along_down_indices) >=3)
    except:
        print(int(centre[0]), int(centre[1]), nearby_along_down_indices)
    #
    nearby_list.append((int(centre[0]), int(centre[1]), nearby_along_down_indices))


#export dataframe
df_nearby = pd.DataFrame(nearby_list)
df_nearby.rename(columns={2:'nearby'}, inplace=True)

df_out = pd.merge(df_tile_params, df_nearby.filter(['nearby']), left_index=True, right_index=True)
df_out.to_csv(os.path.join(output_dir, "tile_adjacencies.csv"), index=False)





