import numpy as np
import pickle
import os
import pandas as pd
import sys

def build_output_list(df_tile_centres,  pop=True):
    nearby_list = []
    # Loop through tile centres, finding nearby patches
    for centre in df_tile_centres.values:
        nearby_along_down_indices = get_neighbour_indices(df_tile_centres, centre, pop)
        # if pop:
        #     nearby_along_down_indices.pop (nearby_along_down_indices.index((int(centre[0]), int(centre[1]))))
        # Add to dictionary, with index of "centre" tile as key
        #nearby_dic[centre_indices] = nearby_along_down_indices
        try:
            assert(len(nearby_along_down_indices) <=8)
            assert(len(nearby_along_down_indices) >=2) #in patches, we can have fewer neighbours
        except:
            print(int(centre[0]), int(centre[1]), nearby_along_down_indices)
            raise
        nearby_list.append((int(centre[0]), int(centre[1]), nearby_along_down_indices))
    return nearby_list


def get_neighbour_indices(df_tile_centres, centre, pop=True):
    # centre_indices = (int(centre[0]), int(centre[1]))
    # Coordinates of tile centre
    cen_xyz = centre[2:5]
    # print('cen_xyz', cen_xyz,  centre)
    # Relative positions of all other tile centres
    # difference_vectors = df_tile_centres.filter(['cen_x','cen_y','cen_z']).values - cen_xyz
    difference_vectors = df_tile_centres.values[:, 2:5] - cen_xyz
    # Distances to other tile centres
    # distances = np.linalg.norm(difference_vectors, axis=1)
    # Select only tile centres below search radius from tile of interest
    # Minimum distance of 10 m to avoid selecting the tile centre (cannot be adjacent to itself)
    #nearby_indices = np.where(np.logical_and(distances < search_radius, distances > 1.))[0]
    #nearby = df_tile_centres.values[nearby_indices, :]
    # Relative indicesall other tiles
    dip_diffs = df_tile_centres.values[:, 0:1] - int(centre[0])
    strike_diffs = df_tile_centres.values[:, 1:2] - int(centre[1])
    adjacent_tiles   = np.logical_and(
                        abs(dip_diffs) <=1, 
                        abs(strike_diffs) <=1)
    # adjacent_tiles   = abs(dip_diffs) <=1
    nearby_indices = np.where(adjacent_tiles)[0]
    nearby = df_tile_centres.values[nearby_indices, :]
    # Relative positions of nearby tiles
    nearby_differences = difference_vectors[nearby_indices, :]
    # print(nearby_indices)
    # Azimuths of vectors to nearby tiles
    angles = np.degrees(np.arctan2(nearby_differences[:, 0], nearby_differences[:, 1]))
    angles[angles < 0] += 360
    # Sort adjacent tiles (currently anticlockwise from due east, I think)
    sorted_nearby = nearby[angles.argsort(), :]
    # Get indices of adjacent tiles
    result = [(int(along), int(down)) for along, down in zip(sorted_nearby[:, 0], sorted_nearby[:, 1])]
    if pop:
        result.pop(result.index((int(centre[0]), int(centre[1]))))
    return result


def find_centre_tile_idx(location_xyz):
    index = tuple(find_centre_tile(location_xyz)[:, 0:2][0])
    return (int(index[0]), int(index[1]))

def find_centre_tile(df, location_xyz):
    difference_vectors = df.filter(['cen_x','cen_y','cen_z']).values - location_xyz
    distances = np.linalg.norm(difference_vectors, axis=1)
    minpos = np.where(distances == np.amin(distances))[0]
    return df.iloc[minpos, :].values   

def get_neighbour_tiles(df_tile_centres, centre, pop=True):
    df = df_tile_centres 
    indices = get_neighbour_indices(df, centre, pop)
    # print ("indices", indices)
    df_tmp = df_tile_centres.copy()
    df_tmp['idx'] = pd.Series(pd.Series([(x[0], x[1]) for x in  df_tile_centres[['along_strike_index', 'down_dip_index']].values]))
    return df[df_tmp['idx'].isin(indices)]


def export(df_tile_centres, df_tile_params):
    nearby_list = build_output_list(df_tile_centres)
    export_data(nearby_list, df_tile_params)


def export_data(nearby_list, df_tile_params):
    #export dataframe
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
    arg2 = "1509514.072933663, 5257251.864946562, -44496.09089646955"

    if arg1 == 'ALL':
        export(df_tile_centres, df_tile_params)
    elif arg1 == 'FIND':
        loc =  tuple([float(x) for x in arg2.split(',')])
        centre = find_centre_tile(df_tile_centres, loc)[:, :][0]
        # print(centre)
        neighbours = get_neighbour_tiles(df_tile_centres, centre, False) #no popping]
        # print(neighbours)
        patch_list = build_output_list(neighbours)
        # for n in patch_list:
        #     print(n)
        export_data(patch_list, df_tile_params)
    
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
