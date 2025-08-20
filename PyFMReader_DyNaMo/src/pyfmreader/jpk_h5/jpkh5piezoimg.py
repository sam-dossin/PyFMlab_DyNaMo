#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 18:43:17 2025

@author: yogehs
"""

from .parsejpkh5header import parsejpkh5_header
import matplotlib.pyplot as plt
import h5py
import numpy as np



def loadJPKimg_h5(filemetadata):
    """
    Returns the contents of the data-image file inside the JPK h5 file.
    the H5 file already has the arrays with the proper valuee.

            Parameters:
                    UFF (uff.UFF): UFF object containing the JPK file metadata.

            Returns:
                    imagedata (dict): dictionary containing all the channels data.
    """
    file_type = filemetadata['file_type']
    image_paths = filemetadata['image_path_dict']
    file = filemetadata['file_path']

    if file_type in 'JPK MultiScan Force Spectroscopy':
        return

    data = {}
    with h5py.File(file, "r") as h5file:

        for name, path in image_paths.items():

            image_arr = h5file[path][:]

            image_2D = image_arr.reshape(
                (filemetadata["num_y_pixels"], filemetadata["num_x_pixels"]))
            data[name] = image_2D

    return data


def computeJPKPiezoImg_h5(UFF, pizeo_channel='MeasuredHeight'):
    """
    Function used to compute the piezo image of a JPK file.

            Parameters:
                    UFF (uff.UFF): UFF object containing the JPK file metadata.

            Returns:
                    piezoimg (np.array): 2D array containing the piezo image of the JPK file.
    """
    file_path = UFF.filemetadata['file_path']

    file_type = UFF.filemetadata['file_type']
    if file_type =='JPK MultiScan Force Map Spectroscopy':
        pizeo_channel = 'CombinedHeightMeasured'
    else:
        print("what hell is the channel name here ", file_type)
    image_path = UFF.filemetadata['image_path_dict'][pizeo_channel]
    with h5py.File(file_path, "r") as h5file:

        tempiezoimg = h5file[image_path][:]
        # TODO is this needed
        piezoimg = tempiezoimg - np.nanmin(tempiezoimg)
        # Reshape piezo image
        piezoimg = piezoimg.reshape(
            (UFF.filemetadata["num_y_pixels"], UFF.filemetadata["num_x_pixels"]))


    return piezoimg


