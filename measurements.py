# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:57:54 2024

@author: taylo
"""

import numpy as np
from scipy.ndimage import center_of_mass, label

def find_beam_waist(probe, pixel_size, criteria='1/e'):
    crit = {
        'FWHM': 0.5,
        '1/e': 1/np.exp(1),
        '1/e^2': 1/np.exp(2),
        }
    probe = np.abs(probe)
    index = np.unravel_index(np.argmax(probe),(probe.shape[0],probe.shape[1]))[0]
    waist = len(probe[index][probe[index] >= crit[criteria]*probe.max()])*pixel_size
    print(f'The beam size ({criteria}) is {waist*10**6:.1f} um')
    return waist
def center_of_mass_per_frame(images):
    cm = []
    for image in np.copy(images):
        image[image < np.average(image)+ np.std(image)] = 0 #need to create an image with zeros where there is no signal
        lbl = label(image)[0] #labels where the image has signal
        cm.append(center_of_mass(image))
    return np.array(cm)

def check_similarities(images):
    """
    Checks the similarities of adjacient images through cross-correlation. 

    Parameters
    ----------
    images : np.ndarray
        Set of diffraction patterns.

    Returns
    -------
    similarities : list
        The correlations of the adjacent images.

    """
    num_images = len(images)
    similarities = []
    for i in range(num_images-1):
        cc = cross_correlate(images[i], images[i+1])
        similarities.append(np.max(np.abs(cc)))
    return similarities 

def cross_correlate(image1,image2):
    """
    Performs a cross-correlation between two images use the Fourier method. 

    Parameters
    ----------
    image1 : np.ndarray
        Reference image.
    image2 : np.ndarray
        Image to scan.

    Returns
    -------
    cross_correlation : np.ndarray
        The cross-correlation between the images.

    """
    image1 = (image1 - np.mean(image1)) / np.std(image1)
    image2 = (image2 - np.mean(image2)) / np.std(image2)    
    I1 = np.fft.fft2(image1)
    I2 = np.fft.fft2(image2)
    cross_correlation = np.fft.fftshift(np.fft.ifft2(np.conjugate(I1)*I2))
    norm_factor = np.sqrt(np.sum(np.abs(image1)**2) * np.sum(np.abs(image2)**2))
    cross_correlation /= norm_factor
    return cross_correlation

if __name__ == '__main__':
    None