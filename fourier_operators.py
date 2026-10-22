# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:36:46 2024

@author: tbuckway
"""

from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq
import numpy as np
from scipy.ndimage import center_of_mass, shift
from . import centering as cen

def prop_fft(obj,center=True,center_method='CM'):
    if center == True:
        r,c = cen.find_center(np.abs(obj),method=center_method)
        shift_y, shift_x  = (obj.shape[-1]-1)/2-np.array([r,c])
        u = fftfreq(obj.shape[-1])
        v = u 
        u,v = np.meshgrid(u,v)
        ramp = phase_ramp(u,v,shift_x,shift_y)
        fftobj = np.fft.fft2(obj)
        obj = np.fft.ifft2(fftobj*ramp).real.squeeze()
    return ifftshift(fft2(fftshift(obj),norm='ortho'))
def inverse_prop_fft(obj,center=True,center_method='CM'):
    if center == True:
        r,c = cen.find_center(np.abs(obj),method=center_method)
        shift_y, shift_x  = (obj.shape[-1]-1)/2-np.array([r,c])
        u = fftfreq(obj.shape[-1])
        v = u 
        u,v = np.meshgrid(u,v)
        ramp = phase_ramp(u,v,shift_x,shift_y)
        fftobj = np.fft.fft2(obj)
        obj = np.fft.ifft2(fftobj*ramp).real.squeeze()
    return fftshift(ifft2(ifftshift(obj),norm='ortho'))

def fresnel_prop(obj, wavelength, z, det_pixel):
    """ Compute Fresnel propagation. """
    k = 2 * np.pi / wavelength
    N = obj.shape[0]  # Number of pixels (assuming square image)
    
    # Create spatial frequency arrays
    dx = det_pixel  # Pixel size in meters
    x = np.fft.fftfreq(N, d=dx)
    y = x
    X, Y = np.meshgrid(x, y)
    
    # Spatial frequencies
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    fy = fx
    FX, FY = np.meshgrid(fx, fy)

    # Compute the quadratic phase term in spatial domain
    H = np.exp(1j * k * z) * np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
    
    # Apply the Fresnel propagation
    obj_freq = prop_fft(obj,center=False)  # Fourier transform of the object
    propagated_freq = obj_freq * H  # Apply phase shift
    propagated = inverse_prop_fft(propagated_freq,center=False)  # Inverse Fourier transform to get back to spatial domain
    
    return propagated

def center_image(image):
    """Deprecated"""
    r,c = center_of_mass(np.abs(image))
    # print(r,c)
    imshape = image.shape[0] #assuming square images.
    shift_y, shift_x  = imshape//2-np.array([r,c])
    # Use scipy.ndimage.shift to apply subpixel shifts
    shifted_image = shift(image, shift=(shift_y, shift_x), order=3)  # Order=3 for cubic interpolation
    return shifted_image

def phase_ramp(u,v,dx,dy):
    ramp = np.exp(-1j*2*np.pi*(u*dx+v*dy))
    return ramp[np.newaxis,:,:]



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import probe_create as pc 
    import plot
    from measurements import find_beam_waist
    from scipy.ndimage import gaussian_filter
    import load
    probegen = pc.probes(29e-9, 13.5e-6,256)
    beam  = probegen.Hermite_Gaussian_beam(l=0, m=0, wo=0.001, z=0.001,w=0.002,xo=0,yo=0)
    probegen.aperture(600e-6)
    # plot.plot_phase(beam)
    # probe.aperture(5.5e-3)
    # probegen.donut_mask(0.5*1000e-6, 0.5*1200e-6)
    # probegen.donut = gaussian_filter(probegen.donut, 0.5)
    # plot.plot_phase(probegen.donut*beam)
    # notcentered= prop_fft(probegen.donut,False)
    # plot.plot_phase(notcentered)
    

    
 