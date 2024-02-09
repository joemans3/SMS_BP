'''
This module contains functions for generating fractional brownian motion samples and computing the mean squared displacement

Functions:
----------
get_fbm_sample(l=1,h=0.5,d=1,n=1)
    Generates a sample of fractional brownian motion
compute_msd_np(xy, t, t_step)
    Computes the mean squared displacement for a given sample

Author: Baljyot Singh Parmar

'''

import numpy as np
from fbm import FBM


def get_fbm_sample(l=1,h=0.5,d=1,n=1):
    '''
    Generates a sample of fractional brownian motion 
    Theory: https://en.wikipedia.org/wiki/Fractional_Brownian_motion
    Implementation is using the fbm package: https://pypi.org/project/fbm/
    Default values are for testing purposes only
    

    Parameters:
    -----------
    l : float,int
        end time (from 0)
    h : float,int  (0 < h < 1)
        hurst parameter, must be between 0 and 1
    d : int 
        dimensions (x,y,z .... ) for one realization, must be greater than 0
    n : int
        even intervals from 0,l, must be greater than 0
    
    Returns:
    --------
    list of lists of numpy arrays, where the first list is the time values for each sample, and the second list is the samples themselves

    Raises:
    -------
    TypeError
        If any of the parameters are not of the correct type
    ValueError
        If any of the parameters are not of the correct value

    Notes:
    ------
    1. The number of samples is equal to the number of dimensions
	'''

    #in the following checks we make sure to print the value of the parameter that is incorrect
    if not isinstance(l, (float,int)):
        raise TypeError("Please enter a valid type for length parameter, you entered: ", type(l))
    if not isinstance(h, (float,int)):
        raise TypeError("Please enter a valid type for hurst parameter, you entered: ", type(h))
    if not isinstance(d, int):
        raise TypeError("Please enter a valid type for dimensions parameter, you entered: ", type(d))
    if not isinstance(n, (float,int)):
        raise TypeError("Please enter a valid type for intervals parameter, you entered: ", type(n))
    if l < 1:
        raise ValueError("Please enter a valid value for length parameter, you entered: ", l)
    if h < 0 or h > 1:
        raise ValueError("Please enter a valid value for hurst parameter, you entered: ", h)
    if d < 1:
        raise ValueError("Please enter a valid value for dimensions parameter, you entered: ", d)
    if n < 1:
        raise ValueError("Please enter a valid value for intervals parameter, you entered: ", n)

    f = FBM(length = l, hurst = h,n = n)
    samples = []
    sample_t = []
    for i in range(d):
        fbm_sample = f.fbm()
        t_values = f.times()  
        if n == 1 and n != l:
            samples.append(fbm_sample[1:])
            sample_t.append(t_values[1:])
        else:
            samples.append(fbm_sample[:-1])
            sample_t.append(t_values[:-1])
    return [sample_t[0], samples]