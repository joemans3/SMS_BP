import numpy as np
def _refecting_boundary(fbm_store_last:float, fbm_candidate:float, space_lim:np.ndarray):
    '''Reflecting boundary condition for the FBM 1D

    Parameters:
    -----------
    fbm_store_last : float
        Last value of the FBM
    fbm_candidate : float
        Candidate value of the FBM
    space_lim : np.ndarray
        Space limit (min, max) for the FBM
    
    Returns:
    --------
    float
        New value of the FBM
    '''
    if fbm_candidate > space_lim[1]:
        #if the candidate is greater than the space limit then reflect the difference back into the space limit
        return space_lim[1] - np.abs(fbm_candidate - space_lim[1])
    elif fbm_candidate < space_lim[0]:
        #if the candidate is less than the negative space limit then reflect the difference back into the space limit
        return space_lim[0] + np.abs(fbm_candidate - space_lim[0])
    else:
        return fbm_candidate