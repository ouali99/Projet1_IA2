import pandas as np
from scipy.spatial import distance

def manhattan(v1, v2) :
    '''
    this function cumpute the Euclidean distance
    v1 and v2 of the same size

    Args :
        v1(array of list): array of the first object
        v2(array of list): array of the second object
    '''
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1-v2))
    return dist 

def euclidean(v1, v2) :
    '''
    this function cumpute the Chebyshev distance
    v1 and v2 of the same size

    Args :
        v1(array of list): array of the first object
        v2(array of list): array of the second object
    '''
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum((v1-v2)**2))
    return dist

def chebyshev(v1, v2) :
    '''
    this function cumpute the Manhattan / citybloc distance
    v1 and v2 of the same size

    Args :
        v1(array of list): array of the first object
        v2(array of list): array of the second object
    '''
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1-v2))
    return dist
def canberra(v1, v2) :
    '''
    this function compute the Canberra distance
    v1 and v2 of the same size

    Args :
        v1(array of list): array of the first object
        v2(array of list): array of the second object
    '''
    distance.ma
    return distance.canberra(v1, v2)