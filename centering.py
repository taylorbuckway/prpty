# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:01:04 2025

@author: taylo
"""

import numpy as np
from scipy.ndimage import center_of_mass, label
from skimage.registration import phase_cross_correlation
import inspect

# --- Router function ---
def find_center(image, method='CM', *, strict=False, **kwargs):
    """
    Route to one of the center-finding methods.

    Parameters
    ----------
    image : any
        The image-like input shared by all methods.
    method : {'CM', 'maximum', 'correlation'}
        Which method to use. Case-insensitive.
    strict : bool, optional (default: False)
        If True, raise an error when kwargs contain names that are not accepted
        by the selected method (other than 'image').
    **kwargs :
        Additional keyword arguments intended for the selected method.

        - with_CM(image) -> no extra kwargs
        - with_maxvalue(image) -> no extra kwargs
        - with_correlation(image, template, mode='valid') -> accepts 'template' (required), 'mode' (optional)

    Returns
    -------
    Any
        The selected method's return value.
    """
    methods = {
        'cm': with_CM,
        'maximum': with_maxvalue,
        'correlation': with_correlation,
    }

    key = str(method).lower()
    if key not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(methods)}")

    func = methods[key]
    sig = inspect.signature(func)

    # Build kwargs compatible with the selected function (excluding 'image')
    allowed_params = {
        name: param
        for name, param in sig.parameters.items()
        if name != 'image'
    }

    # Validate required parameters and collect provided ones
    provided = {}
    missing_required = []
    for name, param in allowed_params.items():
        if name in kwargs:
            provided[name] = kwargs[name]
        elif param.default is inspect._empty:
            # Required and not provided by caller
            missing_required.append(name)

    if missing_required:
        req_list = ', '.join(missing_required)
        raise TypeError(f"{func.__name__}() missing required argument(s): {req_list}")

    if strict:
        # Error on any unexpected kwarg (that the method wouldn't accept)
        unexpected = set(kwargs) - set(allowed_params)
        if unexpected:
            raise TypeError(
                f"Unexpected argument(s) for {func.__name__}: {', '.join(sorted(unexpected))}"
            )

    # Call the function with image and any matched kwargs
    return func(image, **provided)

def with_correlation(image,upsample_factor=100,ref_image=None):
    """
    finds the central part of the data via correlation methods. This works well
    for diffraction patterns with assymetry. Does poorly for diffraction patterns
    with non central zero order scatter (unscatttered light). e.g. donut beams 
    This should be more ideal for beams that appear less clearly. 
    
    Parameters
    ----------
    image : ndarray
        DESCRIPTION.

    Returns
    -------
    r : int
        y pixel location.
    c : int
        x pixel location.

    """
    # autocorrelation is performed and is used as a reference image
    if ref_image is None:
        fft_image = np.fft.fft2(image)
        power_spectrum = np.abs(fft_image)**2/np.max(np.abs(fft_image))
        ref_image = np.fft.fftshift(np.fft.ifft2(power_spectrum).real)
    
    #get the cross correlation
    # F = np.fft.fft2(image)
    # G = np.fft.fft2(autocorrelation)
    # cross_correlation = np.fft.fftshift(np.fft.ifft2(F*np.conjugate(G))).real
    
    # #find max value of cross correlation and shift the image to the center of the array
    # r,c = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
    
    shift, error, phasediff = phase_cross_correlation(reference_image=ref_image,
                            moving_image=image, upsample_factor=upsample_factor,)
    shift_y,shift_x = shift
    return shift_y, shift_x 

def with_CM(image):
    """
    Finds the central part of the data using center of mass from scipy.ndimage
    This is effect for diffraction patterns where the unscatter light has a 
    shape that is non central. e.g. donut beam, plateu like unscatter light.
    It also does not work well for diffraction pattern whose intensity 
    distribution may not be symmetric. This works for symetric pattern.

    Parameters
    ----------
    image : ndarray
        DESCRIPTION.

    Returns
    -------
    r : int
        y pixel location.
    c : int
        x pixel location.

    """
    image[image < np.average(image)+ 2.0*np.std(image)] = 0 #need to create an image with zeros where there is no signal
    lbl = label(image)[0] #labels where the image has signal
    r,c = center_of_mass(image,lbl) #only does center of mass where there is signal
    shift_y, shift_x  = (image.shape[-1]-1)/2-np.array([r,c])
    return shift_y, shift_x

def with_maxvalue(image):
    """
    Simply just finds the location of the maximum signal in the data. This is 
    good for many diffraction patterns whose highest value is at the center of the pattern

    Parameters
    ----------
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    r : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.

    """
    r,c = np.unravel_index(np.argmax(image), image.shape)
    shift_y, shift_x  = (image.shape[-1]-1)/2-np.array([r,c])
    return shift_y, shift_x 
    
    
if __name__ == '__main__':
    import probe_create as pc
    import matplotlib.pyplot as plt
    from scipy.ndimage import shift
    probegen = pc.probes(30e-9, 0.3e-6, 512)
    probe1 = probegen.Hermite_Gaussian_beam(1, 1, 10e-6, 0.001,Eo=1000,xo=10e-6,yo=5e-6)
    probe2 = probegen.Hermite_Gaussian_beam(1, 1, 10e-6, 0.001,Eo=1000)
    probegen.donut_mask(5e-6, 10e-6,offset=[10e-6,5e-6])
    donut1 = probegen.donut 
    probegen.donut_mask(5e-6, 10e-6)
    donut2 = probegen.donut  
    probegen.aperture(diameter=20e-6)
    aper = probegen.aper
    image = np.abs(probe1)**2
    ref_image = np.abs(probe2)**2
    plt.figure(), plt.imshow(donut1)
    
    shift_values = find_center(donut1, method='correlation',upsample_factor=200, ref_image=donut2)      
    print(shift_values)
    
    shifted_image = shift(donut1,shift_values)
    
    plt.figure(), plt.imshow(shifted_image)
    
