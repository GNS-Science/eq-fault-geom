# import pickle
import numpy as np
import os
import pandas as pd
import sys

''' 
def _testing():
    # print(neighbours)
    #add the tuple index
    # 
    print ( df_tile_centres.values[:, 2:5] )

    _411_xyz = (1509514.072933663, 5257251.864946562, -44496.09089646955)

    _410_xyx = (1516532.2337083307,5251172.102316907,-40860.319477146644)
    """
    print (find_centre_tile_idx(_411_xyz)) #.filter(['along_strike_index', 'down_dip_index']))
    print  ()
    print( find_centre_tile(_411_xyz)[:, :][0])
    print(get_neighbour_indices(find_centre_tile(_411_xyz)[:, :][0]))
    """
    centre = find_centre_tile(df_tile_centres, _410_xyx)[:, :][0]

    print('4,10', get_neighbour_indices(df_tile_centres, centre))
    print()

    neighbours = get_neighbour_tiles(df_tile_centres, centre)
    # print(neighbours)
    #print(df_tile_centres)
    print(build_output_list(neighbours))
'''

#for expediency the original distance-based approach was replaced with a index-based approach (REVIEW!) 
def get_neighbour_indices(df_tile_centres, centre):
    """find indices of neighbouring tiles given a central tile :centre:.
    
    :param df_tile_centres: pandas dataframe of all the tile centres (create with subduction_interface)
    :param centre: the central tile as a numpy array
    :return: list of adjacent index pairs
    """

    MAX_IDX_OFFSET = 1
    # Coordinates of tile centre
    cen_xyz = centre[2:5]
    # Relative positions of all other tile centres
    # difference_vectors = df_tile_centres.filter(['cen_x','cen_y','cen_z']).values - cen_xyz
    difference_vectors = df_tile_centres.values[:, 2:5] - cen_xyz

    # filter tiles with index offset indicesall other tiles
    dip_offset = df_tile_centres.values[:, 0:1] - int(centre[0])
    strike_offset = df_tile_centres.values[:, 1:2] - int(centre[1])
    adjacent_tiles   = np.logical_and(
                        abs(dip_offset) <= MAX_IDX_OFFSET, 
                        abs(strike_offset) <= MAX_IDX_OFFSET)

    # adjacent_tiles
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
    result = [(int(along), int(down)) for along, down in zip(sorted_nearby[:, 0], sorted_nearby[:, 1])]
    
    #remove the centre tile index
    result.pop(result.index((int(centre[0]), int(centre[1]))))
    return result


def get_neighbours_distance(tile_centres, centre):

    # We'll need to experiment with this value, should be less than 2e4 for single-tile radius
    search_radius = 1.4e4
    
    # Coordinates of tile centre
    cen_xyz = centre[2:]
    # Relative positions of all other tile centres
    difference_vectors = tile_centres.values[:, 2:] - cen_xyz

    # Distances to other tile centres
    distances = np.linalg.norm(difference_vectors, axis=1)
    # Select only tile centres below search radius from tile of interest
    # Minimum distance of 10 m to avoid selecting the tile centre (cannot be adjacent to itself)
    nearby_indices = np.where(np.logical_and(distances < search_radius, distances > 10.))[0]
    nearby = tile_centres.values[nearby_indices, :]

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
    # centre_indices = (int(centre[0]), int(centre[1]))

    return nearby_along_down_indices    


# def find_centre_tile_idx(location_xyz):
#     index = tuple(find_centre_tile(location_xyz)[:, 0:2][0])
#     return (int(index[0]), int(index[1]))

def find_centre_tile(df, location_xyz):
    """find the centre tile as dataframe, given xyz coords

    :param df_tile_centres: pandas dataframe of tile centres
    :param centre: the central tile as location tuple (x,y,z)

    :return: tile
    """
    difference_vectors = df.filter(['cen_x','cen_y','cen_z']).values - location_xyz
    distances = np.linalg.norm(difference_vectors, axis=1)
    minpos = np.where(distances == np.amin(distances))[0]
    return tuple(df.iloc[minpos, :].values.tolist()[0])

def get_neighbour_tiles(df_tile_centres, centre):
    """build a new dataframe containing tiles and their neighbouring tile indices
    
    :param df_tile_centres: pandas dataframe of tile centres
    :param centre: the central tile as a numpy array

    :return: list
    """

    #get the indices of neighbor tiles
    indices = get_neighbour_indices(df_tile_centres, centre)
    
    #populate an 'idx' column containing the neighbour indices
    df_tmp = df_tile_centres.copy()
    df_tmp['idx'] = pd.Series(pd.Series([(x[0], x[1]) for x in  df_tile_centres[['along_strike_index', 'down_dip_index']].values]))
    
    #return just the rows that have indices (could do this with inner join also)
    return df_tile_centres[df_tmp['idx'].isin(indices)]

def output_list(df_tile_centres):
    """generator of tile indices, and their neighbouring tile indices
    
    :param df_tile_centres: pandas dataframe of tile centres
    :yield: tuple
    """
    # Loop through tile centres dataframe, finding nearby patches
    for centre in df_tile_centres.values:
        yield (int(centre[0]), int(centre[1]),                 # idx tuple
            get_neighbour_indices(df_tile_centres, centre))    # list of neighbour indices

def export(nearby_list, df_tile_params):
    """merge list with params dataframe, write to stdout in csv format

    :param nearby_list: 
    :param df_tile_params: pandas dataframe of tile parameters  
    
    :return: None
    """
    df_nearby = pd.DataFrame(nearby_list).rename(columns={2:'neighbours'})
    #df_nearby, inplace=True)
    cols = list(df_tile_params.columns.values) + ['neighbours']
    df_out = pd.merge(df_tile_params, df_nearby, how='inner', left_on=['along_strike_index', 'down_dip_index'], right_on=[0, 1] )
    df_out .filter(cols) \
        .to_csv(sys.stdout, index=False)

if __name__ == '__main__':

    # data_dir = "/Users/arh79/PycharmProjects/eq-fault-geom/data/"
    data_dir = os.getcwd() #os.path.expanduser("~/DEV/GNS/eq-fault-geom/data/")

    df_tile_centres = pd.read_csv(os.path.join(data_dir, "tile_centres_nztm.csv"), header=0)
    df_tile_params =  pd.read_csv(os.path.join(data_dir, "tile_parameters.csv"), header=0)

    arg1 = "FIND"
    arg2 = "1516532.2337,5251172.102,-40860.0"
    # arg2 = "1509514.072933663, 5257251.864946562, -44496.09089646955"

    if arg1 == 'ALL':
        export(output_list(df_tile_centres), df_tile_params)
    elif arg1 == 'FIND':
        #unpack location 
        loc =  tuple([float(x) for x in arg2.split(',')])
        centre = find_centre_tile(df_tile_centres, loc)

        neighbours = get_neighbour_tiles(df_tile_centres, centre)
        
        patch_list = output_list(neighbours)
        export(patch_list, df_tile_params)
    

