'''Geometric functions for changing patches / coordinate systems'''
# Import libraries
import numpy as np
import tensorflow as tf

###########################################################################
# Coordinate change functions
def CoordChange_C5R10(points, inverse=False):
    """
    Vectorized coordinate change between complex and real representations. 
    Real coordinates arranged: (x1, x2, ..., y1, y2, ...).
    
    Args:
        points (np.ndarray): 
            - If `inverse=False`: array of shape (batch_size, n_complex), dtype complex.
            - If `inverse=True`: array of shape (batch_size, 2 * n_complex), dtype float.
        inverse (bool): Direction of transformation.

    Returns:
        np.ndarray:
            - If `inverse=False`: shape (batch_size, 2 * n_complex), dtype float.
            - If `inverse=True`: shape (batch_size, n_complex), dtype complex.
    """
    if not inverse:
        #return np.stack((point.real, point.imag), axis=-1).reshape(points.shape[0], -1) #...form of (x1, y1, x2, y2, ...)
        return np.concatenate((np.real(points), np.imag(points)), axis=1)
    else:
        #return point[:,::2] + 1j * point[:,1::2] #...form of (x1, y1, x2, y2, ...)
        return points[:, :points.shape[1] // 2] + 1j * points[:, points.shape[1] // 2:]


###########################################################################
# Patch transformation functions
def PatchChange_Coords(coords, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return coords
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return coords ###
    
def PatchChange_G2form(coords, forms, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return forms
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return forms
    
def PatchChange_G2metric(coords, metrics, input_patch=0, output_patch=0):
    if input_patch == output_patch:
        return metrics
    else:
        ### write this (patches labelled 0-4 with 0 the first patch used as input)
        return metrics
    
