# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:25:06 2023

@author: tbuckway
"""

import numpy as np
from scipy.ndimage import center_of_mass, label
from prpty import centering as cen
import gc
import h5py
import psutil
from dataclasses import dataclass
from typing import Union
from lmfit import Model 
import inspect
import matplotlib.pyplot as plt
import warnings

try:
    import cupy as cp
    from cupyx.scipy.ndimage import shift,  gaussian_filter, median_filter
    cp.asarray([1,1])
    xp = cp
    GPU_available = True 
except:
    from scipy.ndimage import shift, gaussian_filter, median_filter
    xp = np
    GPU_available = False

from skimage.transform import resize

class dataprocessing:
    def __init__(self,data,background=None,singlefile=False):
        """
        Initializer of the dataprocessing class. This sets up the data for processing

        Parameters
        ----------
        data : array[P,M,N]
            DESCRIPTION.
        background : TYPE, optional
            DESCRIPTION. The default is None.
        singlefile : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        self.data = np.asarray(data) 
        self.originaldata = np.copy(self.data)
        try:
            self.background = np.asarray(background)
        except ValueError:
            self.background = background
        self.singlefile = singlefile
    def backgroundsub(self,background_process=None):
        if self.singlefile == True:
            self.data = self.data - self.background
        elif background_process == 'average':
            self.data = np.sum(self.data-np.sum(self.background,0)/len(self.background),0)
        elif background_process == 'sum':
            self.data = np.sum(self.data,0)-np.sum(self.background,0)
        else:
            self.data = self.data-self.background
        return self.data
    def crop(self,cropto=512,backinc=False):
        x,y = np.where(self.data==np.max(self.data))
        self.data = self.data[x[0]-int(cropto/2):x[0]+cropto/2,y[0]-cropto/2:y[0]+cropto/2]
        return self.data
    def FVB(self,rotate = True):
        if rotate == True:
            self.data = np.sum(np.transpose(self.data),0)
        else:
            self.data = np.sum(self.data,0)
        return self.data
    def threshold(self,value=0):
        self.data[self.data<value]=0
        return self.data
    def countspersecond(self, t):
        if self.singlefile==True:
            self.data = self.data/t
        else:
            self.data = self.data/(t*len(self.originaldata))
        return self.data
    def spectrumprocessing(self,background_process,value,t):
        self.backgroundsub('average')
        self.threshold(value=value)
        self.FVB()
        self.countspersecond(t=t)
        self.data
        return self.data
    def countstophotons(self,energy,gain):
        self.photons = self.data*gain*3.65/energy
        return self.photons 
    def countstophotonspersecond(self,energy,gain,t):
        self.countspersecond(t)
        self.countstophotons(energy, gain)  
        return self.photons 

class imageprocessing:
    def __init__(self,
                 data,
                 background,
                 start_image = None,
                 end_image = None, 
                 every_other=None,
                 back_region=[2000,2048,2000,2048],
                 batch_num=1,
                 pad=0,
                 low=0,
                 high=10,
                 hot_pixel_params = None, #[2*16,None],
                 remove_dead_pixels=False,
                 dead_pixels = None,
                 center_images=False,
                 find_center_method='CM',
                 center_method='shift',
                 threshold=None,
                 tech='gaussian',
                 mask_sigma=None,
                 filter_sigma=5,
                 truncate=4.0,
                 size=5,
                 diff_signal=None,
                 th_median_size=1, # applies median filter to data after thresholding. Applies to entire dataset. 
                 use_median_filter=False, # applies median filter to fixed pixels only and not the whole dataset
                 median_threshold= 0.001,
                 median_size=5
                 ):
        try:
            import prpty.random_utils as rt
        except ModuleNotFoundError:
            import random_utils as rt

        """
        Initializes the image processing class.

        Parameters
        ----------
        data : type(h5py.Dataset)
            The data from ptychographic setup will have shape of (N,M,M) where N is the number of images and M 
            is the size of the data. Data should be given as a h5py.Dataset type. 
        background : np.ndarray
            Background image(s) of the camera. 
        start_image: int or None
            First index in data i.e. (data[start_image:end_image:every_other])
        end_image: int or None
            Last index in data i.e. (data[start_image:end_image:every_other])
        every_other: int or None 
            Skip every n images. i.e. (data[start_image:end_image:every_other])
        back_region: list
            Region in the background used for the minimization method for background substraction see remove_background function
        batch_num: int, optional
            Separates the images into a number of batches to use less memory. The default is 1.
        pad: int, optional
            Pad each image with zeros. This is important when cropping to the full M,N image. Otherwise,
            the images may not be centered properly. The default is 0. Typically not used anymore. 
        Low: int
            Ignore this variable
        high: 
            Ignore this variable
        hot_pixel_params: bool,list, optional
            Sets the parameters for removing hot pixels. Default is None which turns removing hot pixels off. 
        center_images: bool
            Centers the images if True. See center_images function for more details.
        find_center_method: str
            Method used to find center. See center_images function for more details.
        threshold: int or bool
            Thresholds the data see remove_background and remove_noise functions for more details. 
        mask_sigma: bool
            see remove_noise function
        filter_sigma: int
            See remove_noise function
        truncate: float
            see remove_noise function
        size: int
            See remove_noise function
        diff_signal: None or np.ndarray
            See remove_background function
        th_median_size: int
            Size of the kernal for the median filter applied to data after thresholding. see remove_noise
        use_median_filter: bool
            Determines whether a median filter is applied to data. see apply_median_filter
        median_threshold: float
            see apply_median_filter
        median_size: int
            see apply_median_filter
        Returns
        -------
        None

        """ 
        print("Beginning image processing!")
        if type(data) != h5py.Dataset:
            print("PrPty note: It is recommended to input the data as a h5py.Dataset. Processing will continue with the array")
        self.data = np.asarray(data[start_image:end_image:every_other]).astype(np.float64)
        if len(background.shape) > 2:
            self.background = np.asarray(background[0]).astype(np.float64)
        else:
            self.background = np.asarray(background).astype(np.float64)
        self.data_shape = self.data.shape
        self.num_images = self.data.shape[0]
        self.pad = pad
        self.batch_size = np.ceil(self.data.shape[0]/batch_num).astype(int)
        if pad != 0:
            try:
                self.data = rt.random_pad(self.data, ((0,0),(pad,pad),(pad,pad)), low, high)
            except ValueError:
                if len(np.shape(self.data)) == 5:
                    self.data= rt.random_pad(self.data,((0,0),(0,0),(0,0),(pad,pad),(pad,pad)),low,high)
                else:
                    self.data = rt.random_pad(self.data,((pad,pad),(pad,pad)),low,high)
        if hot_pixel_params is not None:
            self.remove_hot_pixels(*hot_pixel_params)
        if remove_dead_pixels:
            self.remove_dead_pixels(dead_pixels)
        if center_images == True:
            self.center_images(find_center_method=find_center_method,center_method=center_method)
        if use_median_filter == True:
            self.apply_median_filter(median_threshold, median_size)
        if diff_signal:
            self.remove_background(back_region=back_region,
                                      method='localized erasing',
                                      diff_signal=diff_signal
                                      )
        if threshold is not None:
            self.remove_background(back_region=back_region)
            self.remove_noise(threshold=np.float64(threshold),tech=tech,mask_sigma=mask_sigma, filter_sigma=filter_sigma,truncate=truncate,size=size,apply_median_size=th_median_size)
    def crop(self,cropto=512,center_method='maximum'):
        """
        Crops image(s) while also center the image to the highest count. 

        Parameters
        ----------
        cropto : int, optional
            Crops the image(s) to this size. The default is 512.
        center_method : str or list
            Method to detemine center of image. Pixel may also be specified as a list

        Raises
        ------
        Exception
            Insufficient padding on the image.

        Returns
        -------
        array (P,cropto,cropto)
            Cropped and centered images.

        """
        print(f'Cropping images to {cropto} using {center_method} method')
        if cropto == self.data_shape[-1]+self.pad:
            return
        if len(self.data.shape) == 2:
            tempdata = np.copy(self.data)
            tempdata[tempdata < np.average(tempdata)+ np.std(tempdata)] = 0 #need to create an image with zeros where there is no signal
            lbl = label(tempdata)[0] #labels where the image has signal
            r,c = np.round(center_of_mass(tempdata,lbl)).astype(int) #only does center of mass where there is signal
            self.data = self.data[r-cropto//2:r+cropto//2,c-cropto//2:c+cropto//2]
        elif len(self.data.shape) == 5:
            tempdata = np.copy(self.data[0,0,0])
            tempdata[tempdata < np.average(tempdata)+ np.std(tempdata)] = 0 #need to create an image with zeros where there is no signal
            lbl = label(tempdata)[0] #labels where the image has signal
            r,c = np.round(center_of_mass(tempdata,lbl)).astype(int) #only does center of mass where there is signal
            self.data = self.data[:,r-cropto//2:r+cropto//2,c-cropto//2:c+cropto//2]
        else:
            dsum = np.average(np.abs(self.data),0)
            try: 
                r,c = cen.find_center(image=dsum,method=center_method)
            except:
                assert type(center_method) == list, 'center_method must be given as list if specifying center pixel for cropping'
                assert len(center_method) == 2, 'List must be of length 2 represented as [row,col]'
                r,c = center_method
            self.data = self.data[:,r-cropto//2:r+cropto//2,c-cropto//2:c+cropto//2]
        self.background = self.background[r-cropto//2:r+cropto//2,c-cropto//2:c+cropto//2]
        if 0 in self.data.shape:
            raise Exception("Insufficient padding of the data in order to center the data. Increase the pad parameter.")
    def binpix(self,binning=2):
        """
        Bins the image(s). Does nothing if binning is 1. 

        Parameters
        ----------
        binning : int, optional
            How much the images will be binned. The default is 2.

        Raises
        ------
        
            The bin number must able to equally divide the image size.

        Returns
        -------
        array (P,M//binning,N//binning)
            Binned images.

        """
        print(f'Binning data by {binning} pixels')
        if binning == 1:
            return self.data
        if self.data.shape[-1]%binning != 0:
            raise  Exception(f'binning by {binning} cannot divide equally array of size {self.data.shape[1:]}')
        if len(self.data.shape) == 2:
            self.data = np.reshape(self.data,(self.data.shape[0]//binning,binning,self.data.shape[1]//binning,binning)).sum(1).sum(-1)
        else:
            self.data = np.reshape(self.data,(self.data.shape[0],self.data.shape[1]//binning,binning,self.data.shape[2]//binning,binning)).sum(2).sum(-1)
        self.background = np.reshape(self.background,(self.background.shape[0]//binning,binning,self.background.shape[1]//binning,binning)).sum(1).sum(-1) 
    def remove_noise(self,threshold,tech='gaussian',mask_sigma=None, filter_sigma=None,size=None,truncate=None,apply_median_size=1):
        """
        Remove noise from diffraction patterns. The noise is removed by creating 
        a filter mask which is a filtered copy of the data. This is what is thresholded
        and then used to set values of the original data to zero. This is an 
        effective method at removing noise while preserving the diffraction 
        signal. 

        Parameters
        ----------
        threshold : None or float
            Threshold to apply to the filter mask.
        tech : str, optional
            The filter used on the mask. Options are 'gaussian' and 'median'. The default is 'gaussian'.
        mask_sigma : float or bool, optional
            Whether to apply a Gaussian filter to the mask for relaxation. The default is None
        filter_sigma : float, optional
            Sigma perameter for the Gaussian filter option. The default is None.
        size : int, optional
            Kernal size parameter for the median filter option. The default is None.
        truncate : float, optional
            How many standard deviations to use for Gaussian filter option. The default is None.
        apply_median_size : int, optional
            Applies a median filter for a given kernal size to the data after thresholding. The default is 1.

        Raises
        ------
        Exception
            f'{tech} is an invalid filter mask. Accepted entries are \'gaussian\' and \'median\'. '.

        Returns
        -------
        None.

        """
        print(f'Removing noise using {tech} threshold mask with threshold of {threshold}')
        for start, end in batch_index_generator(self.num_images, self.batch_size):
            batch = xp.asarray(self.data[start:end])
            if tech == 'median':
                filter_mask = median_filter(batch,size=(1,size,size))
            elif tech == 'gaussian':
                filter_mask = gaussian_filter(batch, sigma=(0,filter_sigma,filter_sigma),truncate=truncate)
            else:
                raise Exception(f'{tech} is an invalid filter mask. Accepted entries are \'gaussian\' and \'median\'. ')
            filter_mask = xp.where(filter_mask < threshold, 0, 1).astype(xp.float64)
            if mask_sigma:
                filter_mask = gaussian_filter(filter_mask,sigma=(0,mask_sigma,mask_sigma),truncate=1.5)
            batch *= filter_mask
            if apply_median_size != 1:
                batch = median_filter(batch,size=(1,apply_median_size,apply_median_size))
            self.data[start:end] = batch if not GPU_available else batch.get()
            del batch, filter_mask
            None if not GPU_available else clear_GPU_memory()
    def remove_background(self,back_region,method='minimization',num_std=2,diff_signal=[0,0,0,0]):
        """
        This removes background noise from the data using three different methods. 
        1) A simple threshold. Useful on uniform backgrounds
        2) Minimization method that takes a background and adapts the substraction
        of it from frame to frame. Very effective on uniform backgrounds.
        3) The localized erasing method which is used for backgrounds that have no
        uniformity to it such as undesired scattering. This only works if there is a constant
        undesired scattered signal in the background. It finds this high noise scatter and removes
        it. 

        Parameters
        ----------
        back_region : list
            The select background region used to remove background shouldn't have a signal
            Given as [row low, row high, column low, colomn high].
        method : string, optional
            Type of threshold method. Accepted entries are 'threshold', 'localized erasing', and
            minimization. The default is 'threshold'.
        num_std : float, optional
            Only applies to the 'threshold method'. How many standard deviation below the average do you want to 
            remove. TSee reference for more information. The default is 2.
        diff_singal : list 
            The region that contains the diffraction pattern that should not be erased. 
            This is applied to the localized erasing method and ensures only the 
            nonuniform localized high noise signal is the only part of the data is 
            erased. Given as [row low, row high, column low, column high]

        Returns
        -------
        None.
        C. Wang, Y. Wang, J. Wang, Z. Xu, R. Tai, and H. Liu, “Background noise removal in x-ray ptychography,” 
            Appl. Opt. Vol. 56, Issue 8, pp. 2099-2111, vol. 56, no. 8, pp. 2099–2111, Mar. 2017, 
            doi: 10.1364/AO.56.002099.

        """
        print(f'Removing background using {method} method')
        rlow, rhigh, clow, chigh = back_region
        IRmax = np.max(self.data[:,rlow:rhigh,clow:chigh],axis=(-1,-2))
        IRstd = np.std(self.data[:,rlow:rhigh,clow:chigh],axis=(-1,-2))
        I_threshold = np.average(IRmax)-2*np.average(IRstd)
        if method == 'threshold':
            I_threshold = np.average(IRmax)-num_std*np.average(IRstd)
            self.data[self.data<I_threshold] = 0
        elif method == 'localized erasing':
            Nc = np.zeros_like(self.data)
            Nc[self.data>I_threshold] = 1
            Nc = np.sum(Nc,0)
            Nc[Nc<self.data.shape[0]/2] = 0
            drlow, drhigh, dclow, dchigh = diff_signal
            Nc[drlow:drhigh,dclow:dchigh] = 0
            Nc[Nc>0] = 1
            Nc = np.abs(Nc-1)
            self.data *= Nc 
        elif method == 'minimization':
            Im = self.data[:,rlow:rhigh,clow:chigh]
            ID = self.background[rlow:rhigh,clow:chigh]
            alpha = np.sum(Im*ID,axis=(-1,-2))/np.sum(ID**2+1e-12)
            alpha = alpha[:,np.newaxis,np.newaxis]
            back = self.background[np.newaxis,:,:]
            for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                self.data[start:end] -= alpha[start:end]*back #TODO: this uses a lot of RAM memory particularly at the alpha*back. Adjust so it's less
            self.data[self.data<0] = 0
        else:
            raise KeyError(f'{method} is not a valid method. Acceptable methods are: \'threshold\' \'localized erasing\' and \' minimization\' ' )
            
    def im_resize(self,scale=2):
        if len(self.data.shape) != 2:
            raise Exception('Must be a single image not stack')
        self.data = resize(self.data,(self.data.shape[0]*scale,self.data.shape[1]*scale),anti_aliasing=True)
        return self.data 
    def crop_resize(self,cropto=512,scale=2):
        self.data = self.im_resize(scale)
        self.data = self.crop(cropto)
        return self.data
    def center_images(self,find_center_method='CM',center_method='shift'):
        """
        Centers the images subpixelly. The center can be specified or determined. 
        The center method has two options: 'shift' and 'fourier_shift'. The 'shift'
        method uses scipy.ndimage.shift or cupyx.scipy.ndimage.shift and 
        is generally safer but can take a significant amount of time to 
        center. The 'fourier_shift' takes advantage of the shift theorem in Fourier
        transforms. This method is much faster making it ideal for large datasets. Care 
        must be taken because this method may produce ringing effects in the data 
        particularly at hard edges or datasets with a lot of zeros around the signal. 

        Parameters
        ----------
        find_center_method : str, optional
            Defines method to find the center of the diffraction pattern. 
            Methods listed in prpty.centering. A specific pixel may also be 
            specified. The default is 'CM'.
        center_method : str, optional
            Centering method to shift images by subpixels. Options are 'shift' and 'fourier_shift'. The default is 'shift'.

        Raises
        ------
        Exception
            '{center_method} is not a centering method. Options are fourier_shift and shift'.

        Returns
        -------
        None.

        """
        #TODO: include option to pad the images to better shift the image accurately. Even better subpixel registration
        print(f"Centering images using {center_method} method ")
        dsum = np.sum(self.data,0)
        try:
            shift_y,shift_x = cen.find_center(image=dsum,method=find_center_method)
        except:
            assert type(find_center_method) == list, 'find_center_method must be given as list if specifying center pixel. May be subpixel.'
            assert len(find_center_method) == 2, 'List must be of length 2 represented as [row,col]'
            shift_y, shift_x = find_center_method
        # shift_y, shift_x  = (self.data.shape[-1]-1)/2-np.array([r,c])
        print(f"Shifting image by {shift_x} x and {shift_y} y")
        if center_method == 'fourier_shift':
            u = xp.fft.fftfreq(self.data.shape[-1])
            v = u 
            u,v = xp.meshgrid(u,v)
            ramp = phase_ramp(u,v,shift_x,shift_y)
            backbatch = xp.asarray(self.background)
            backfft = xp.fft.fft2(backbatch)
            backbatch = xp.fft.ifft2(backfft*ramp).real.squeeze()
            self.background = backbatch if not GPU_available else backbatch.get()
            del backbatch, backfft, u,v
            None if not GPU_available else clear_GPU_memory()
            for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                batch = xp.asarray(self.data[start:end])
                fftbatch = xp.fft.fft2(batch,axes=(-2,-1))
                batch = xp.fft.ifft2(fftbatch*ramp,axes=(-2,-1)).real
                self.data[start:end] = batch if not GPU_available else batch.get()
                del batch, fftbatch
                None if not GPU_available else clear_GPU_memory()
        elif center_method == 'shift':
            backbatch = xp.asarray(self.background)
            backbatch = shift(backbatch,shift=(shift_y,shift_x),order=1,mode='constant',cval=0.0)
            self.background = backbatch if not GPU_available else backbatch.get()
            del backbatch
            None if not GPU_available else clear_GPU_memory()
            self.background = np.clip(self.background,0,None)
            for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                batch = xp.asarray(self.data[start:end])
                batch = shift(batch, shift=(0,shift_y, shift_x), order=1,mode='constant',cval=0.0)  # Order=3 for cubic interpolation
                self.data[start:end] = batch if not GPU_available else batch.get()
                del batch
                None if not GPU_available else clear_GPU_memory()
            # self.data = np.clip(self.data,0.0,None)
            self.data[self.data<=0.0] = 0 
        else:
            raise Exception(f'{center_method} is not a centering method. Options are fourier_shift and shift')

    def apply_median_filter(self,threshold,size):
        """
        Applies a median filter on data. This will remove low varying pixels which
        come from dust on camera or salt and pepper. This is for fixed pixel salt and pepper noise.
        This method does not work for salt and pepper noise that non fixed pixel noise that varies 
        from frame to frame such as readout noise. See remove_hot_pixel as an alternative for removing hot pixels. 
        
        Parameters
        ----------
        threshold : float
            Threshold value betwen 0 and 1 for exluding low varying background.
        size : int
            See size parameter for the cupyx.scipy.ndimage.median_filter. 

        Returns
        -------

        """
        print(f'Applying median filter on data with threshold of {threshold} and kernal size of {size}')
        ave_image = np.mean(self.data,0)
        med_ave_image = median_filter(ave_image,size=5)
        ave_diff = np.abs(med_ave_image-ave_image)
        ave_diff[ave_diff<threshold*ave_diff.max()] = 0
        mask = np.zeros_like(ave_image)
        mask[ave_diff>0] = 1 
        for image in range(self.data.shape[0]):
            tempdata = np.copy(self.data[image])
            mdata = median_filter(tempdata,size)
            tempdata[mask == 1] = mdata[mask==1]
            self.data[image] = tempdata
    def remove_hot_pixels(self,bitdepth=2**16,pixels=None):
        """
        Removes hot pixels from diffraction patterns by applying a median filter
        on that pixel so it matches values around it. If pixels is specified,
        this will set those pixels to zero. 

        Parameters
        ----------
        bitdepth : int, optional
            The bit depth of each pixel in camera. This uniformly removes the pixels
            without discrimination. 
            The default is 2**16 for a 16-bit camera.
        pixels : list, optional
            List of pixels including the frame location. This method should be 
            used when you may have saturated your camera. You want to be selective
            with the pixels in this case. 
            e.g. [[frame1,rowpixel1,colpixel2],[frame2,rowpixel2,rowpixel3]]. 
            The default is None.

        Returns
        -------
        None.

        """
        print(f"Removing hot pixels with value of {int(bitdepth-1)}")
        if pixels is not None:
            for pixel in pixels:
                frame, row, col = pixel
                self.data[frame,row,col] = 0
                print('Warning! If you saturated the camera, it is recommended')
                print('to retake the data with lower exposure time.')
        else:
            self.data[self.data == bitdepth-1] = 0
            dsum = np.sum(self.data,0)
            for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                batch = xp.asarray(self.data[start:end])
                filtered_batch = gaussian_filter(batch.copy(),sigma=(0.0,1.0,1.0))
                batch[:,dsum == 0] = filtered_batch[:,dsum == 0]
                # batch[batch == bitdepth -1] = filtered_batch[batch == bitdepth-1]
                self.data[start:end] = batch if not GPU_available else batch.get()
                del batch, filtered_batch
                None if not GPU_available else clear_GPU_memory()
                
    def remove_dead_pixels(self,pixels=None):
        """
        Removes dead pixels from diffraction patterns by applying a gaussian filter
        on that pixel so it matches values around it. If pixels are specified,
        it will only use it on those pixels (recommended). If not specified, 
        the function will apply gaussian filter to any zero pixel indiscriminantly.
        If not specified, a sum of the images will be used to find the dead pixels
        which is less likely to show true zero valued pixels. 

        Parameters
        ----------

        pixels : list, optional
            List of pixels. It is assumed these pixels are dead in all frames. 
            e.g. [[rowpixel1,colpixel2],[rowpixel2,rowpixel3]]. 
            The default is None.

        Returns
        -------
        None.

        """
        print("Removing dead pixels")
        if pixels is not None:
            for pixel in pixels:
                row, col = pixel
                for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                    batch = xp.asarray(self.data[start:end])
                    mask_frame = gaussian_filter(batch.copy(),sigma=(0,1.0,1.0))
                    batch[:,row,col] = mask_frame[:,row,col]
                    self.data[start:end] = batch if not GPU_available else batch.get()
                    del batch, mask_frame
                    None if not GPU_available else clear_GPU_memory()
        else:
            print("Warning: Not specifying dead pixels could apply gaussian filter")
            print("to any zero valued pixel which may not be a dead pixel.")
            dsum = np.sum(self.data,0)
            for start, end in batch_index_generator(self.data.shape[0], self.batch_size):
                batch = xp.asarray(self.data[start:end])
                filtered_batch = gaussian_filter(batch.copy(),sigma=(0,1.0,1.0))
                batch[:,dsum == 0] = filtered_batch[:,dsum == 0]
                self.data[start:end] = batch if not GPU_available else batch.get()
                del batch, filtered_batch
                None if not GPU_available else clear_GPU_memory()
    def remove_artifacts(self):
        #TODO: find a way to remove vbleeds and back reflections from data. Diffraction is symmetric so these asymmetric parts could be removed. 
        dave = np.average(self.data,0)
        dave[dave>=dave.max()/np.exp(3.5)] = 0
        self.data -= dave 
class ptychoprocessing:
    def __init__(self,data,positions):
        """
        Processing data to prepare for ptychography. Initializer

        Parameters
        ----------
        data : array
            (P,M,M) Image array to process for ptychography. Should be square images.
        positions : array
            (P,2) Position array corresponding to each image. 
        Returns
        -------
        None.

        """
        self.data = np.asarray(data)
        self.positions = np.asarray(positions) 
    def rm_low_signal_frames(self, threshold):
        """
        Removes low signal frames from ptychography dataset. Ptychography struggles
        with low to no signal frames. 

        Parameters
        ----------
        threshold : float
            Value betwen 0 and 0.99. This takes threshold and times it by the max value
            of the images. This is then used to remove any images that don't have values
            above this threshold*max(data)

        Returns
        -------
        None.

        """
        maxsigperframe = np.max(self.data,axis=(1,2))
        lowsignframes = np.where(maxsigperframe>=threshold*maxsigperframe.max())[0]
        self.data = self.data[lowsignframes]
        self.positions = self.positions[lowsignframes]
    def data_orientations(self, adjust_data=''):
        """
        Image and position orientation adjustments to correct the directions of 
        either the images or positions
        Accepted values: 'v', 'h', 'r-', 'r+', 'xs', 'ys', 'pf'
        Also accepts combinations for example 'vh' which is a vertical and horizontal flip of the images

        Parameters
        ----------
        adjust_data : string, optional
            DESCRIPTION. The default is ''.

        Returns
        -------
        None.

        """
        if 'v' in adjust_data:
            #vertically flip images
            print('Vertically flipping images')
            self.data = np.flip(self.data, 1)
        if 'h' in adjust_data:
            #horizontally flip images
            print('Horizontally flipping images')
            self.data = np.flip(self.data, 2)
        if 'r-' in adjust_data:
            #rotate clockwise
            print('Rotating images clockwise by 90 degrees')
            self.data = np.rot90(self.data,axes=(2,1))
        if 'r+' in adjust_data:
            #rotate counter clockwise
            print('Rotating images counterclockwise by 90 degrees')
            self.data = np.rot90(self.data,axes=(1,2))
        if 'xs' in adjust_data:
            #switches the signs of x
            # self.positions[:,0] = np.flip(self.positions[:,0])
            print('Flipping the x positions')
            self.positions[:,0] = -1*self.positions[:,0]
        if 'ys' in adjust_data:
            #switches the signs of y
            # self.positions[:,1] = np.flip(self.positions[:,1])
            print('Flipping the y positions')
            self.positions[:,1] = -1*self.positions[:,1]
        if 'pf' in adjust_data:
            #flips the x and y coordinates in positions
            print('Switching x and y positions')
            self.positions = np.flip(self.positions,1)
    def break_position_grids(self,x_error=0.0,y_error=0.0,**kwargs):
        """
        Breaks the grid patterns to prevent grid like artifacts in ptychography
        by adding random small errors to each position. Use position refinement
        if breaking symmetry.

        Parameters
        ----------
        x_error : float
            percent error in first column.
        y_error : float
            percent error in second column.

        Returns
        -------
        None.

        """
        unique_x = np.unique(self.positions[:,0])
        unique_y = np.unique(self.positions[:,1])
        
        dx_intervals = np.diff(unique_x)
        dy_intervals = np.diff(unique_y)
        
        dx = np.median(dx_intervals)
        dy = np.median(dy_intervals)
        
        rng = np.random.default_rng()
        rand_adj = np.zeros_like(self.positions)
        rand_adj[:,0] = rng.uniform(low=-dy*y_error,high=dy*y_error,size=self.positions.shape[0])
        rand_adj[:,1] = rng.uniform(low=-dx*x_error,high=dx*x_error,size=self.positions.shape[0])
        self.positions += rand_adj
        
class probe_import_processing:
    def __init__(
            self,
            probe,
            binto,
            num_pix,
            center_point=None,
            ):
        """
        Class for processing imported probes from other reconstructions. This takes a previous reconstructed probe and 
        ensures it matches the probe size and pixel size to prepare for the next reconstruction. It is advised to work with 
        arrays that whose sizes are bases of 2. e.g 32,64,...512...2048.
        
        TODO: Change the class such that instead of binto and cropto changes we give previous resolution
        and sampling and match it with current resolution and sampling. The would be more generic. 

        Parameters
        ----------
        probe : (...,N,M,M) array 
            A probe array that constists of several hermite gaussian probes(N).
        probe_import : str
            This string should contain information about the previous reconstruction
            It looks for anything that as the pattern 'crop:####' and 'bin:##'.
        binto : int
            The reconstructions binned data. This determines how sampled the data will be.
            The imported reconstructed probe will be adjusted to match the same sampling
        cropup : int
            This is the number of pixels used for the reconstruction. This determines the resolution
            of the probe and object. The imported reconstructed probe will be adjusted
            to match the same resolution as the reconstruction. 

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self.probe = probe
        self.binto = binto
        self.num_pix = num_pix
        self.adjust_sampling(center_point)
        self.adjust_resolution()
    def adjust_sampling(self,center_point=None):
        if self.binto > 1:
            r,c = center_point if center_point is not None else (dim // 2 for dim in self.probe.shape[-2:])
            cropto = self.probe.shape[-1]//self.binto
            print('Cropping imported probe to match sampling of data')
            self.probe = self.probe[...,r-cropto//2:r+cropto//2,c-cropto//2:c+cropto//2] 
    def adjust_resolution(self):
        target_shape = self.probe.shape[:-2] + (self.num_pix,self.num_pix)
        amplitude = resize(np.abs(self.probe),target_shape,preserve_range=True)
        phase = resize(np.angle(self.probe),target_shape,preserve_range=True)
        print('Adjusting probe resolution to match size of detector')
        self.probe = amplitude*np.exp(1j*phase)

@dataclass 
class post_processing:
    image: Union[np.ndarray,list]
    def __post_init__(self):
        if not isinstance(self.image, (np.ndarray,list)):
            raise TypeError(
                f"'image' must be {np.ndarray,list}, got {type(self.image)} instead. ")
        if isinstance(self.image, list):
            self.image = np.array(self.image) 
        if isinstance(self.image.dtype, (np.dtypes.Complex64DType,np.dtypes.Complex128DType)):
            self.image_phase = np.angle(self.image)
        else:
            self.image_phase = None
    def cropto(self):
        pass
    def ramp_correction_MG(self,mask=None,window=1.5,upsample=100,
                           n_stage=2, polish=False, verbose=False):
        """
        Correct the linear and constant phase ramps in objects by minimization.
        
        M. Guizar-Sicairos et al., “Phase tomography from x-ray coherent diffractive imaging projections,”
        Optics Express, Vol. 19, Issue 22, pp. 21345-21357, vol. 19, no. 22, pp. 21345–21357, Oct. 2011, 
        doi: 10.1364/OE.19.021345.
  

        Parameters
        ----------
        mask : numpy.ndarray, optional
            Window representing flat region in object with 1's being known flat or vacuum region and 
            0's elsewhere. If not given, a window of the object opens and prompts
            to draw a polygon on image. The default is None.
        window : float, optional
            Window size for around max value for F[mask*image] with F being the Fourier Transform for subpixel
            registration. The default is 1.5.
        upsample : int, optional
            How much to upsample window for subpixel registration. The default is 100.

        Returns
        -------
        image: numpy.ndarry
            Image with constant and phase ramps removed.
        mask : numpy.ndarray, bool
            Mask use for empty region.
        parameters : dict
            Fitted parameters u, v, and a.

        """
        if mask is None:
            self._draw_polygon()
            mask = self._polygon_mask()
            
        f = mask*np.exp(1j*np.angle(self.image))
        row, col = f.shape 
        
        f_pad_twice = np.pad(f,((row//2,row//2),(col//2,col//2)),mode='constant')
        R = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f_pad_twice)))
        u0_twice, v0_twice = np.unravel_index(np.argmax(np.abs(R)), f_pad_twice.shape)
        
        pr, pc = f_pad_twice.shape
        u0 = (u0_twice - pr//2) * row / pr
        v0 = (v0_twice - pc//2) * col / pc
        
        u_best, v_best, a = u0, v0, 0.0
        win, n_pts = window, None 
        steps = []
        for stage in range(max(int(n_stage),1)):
            R_fine, U, V, X, Y = subpixel_image_registration(
                f, (u_best,v_best),window=win,upsample=upsample,n_points=n_pts)
            idx = np.unravel_index(np.argmax(np.abs(R_fine)),R_fine.shape)
            
            u_best, v_best = U[idx], V[idx]
            a = np.angle(R_fine)[idx]
            
            step = U[1,0] - U[0,0]
            steps.append(step) 
            n_pts = U.shape[0] 
            win = 4*step
            
        if polish:
            from scipy.optimize import minimize
     
            def neg_peak(p):
                u, v = p
                ex = np.exp(-2j*np.pi*u*np.arange(row)/row)
                ey = np.exp(-2j*np.pi*v*np.arange(col)/col)
                return -np.abs(ex @ f @ ey)
     
            res = minimize(neg_peak, (u_best, v_best), method='Nelder-Mead',
                           options={'xatol': 1e-10, 'fatol': 1e-12})
            if res.success:
                u_best, v_best = res.x
                ex = np.exp(-2j*np.pi*u_best*np.arange(row)/row)
                ey = np.exp(-2j*np.pi*v_best*np.arange(col)/col)
                a = np.angle(ex @ f @ ey)           
            
        X, Y = np.meshgrid(np.arange(row),np.arange(col),indexing='ij')
        ramp_correction = 2*np.pi*(u_best*X/row+v_best*Y/col) + a
        
        if verbose:
            print(f"coarse (u,v) = ({u0:+.4f}, {v0:+.4f})")
            print(f"final  (u,v) = ({u_best:+.6f}, {v_best:+.6f}), a = {a:+.6f}")
            print(f"grid steps   = {['%.2e' % s for s in steps]}")
            plt.figure(), plt.imshow(ramp_correction, cmap='seismic')
            plt.colorbar(), plt.show(block=False), plt.pause(0.5)
     
        self.image = self.image * np.exp(-1j*ramp_correction)
        if getattr(self, 'image_phase',None) is not None:
            self.image_phase = np.angle(self.image)
        parameters = {
            'u': u_best,
            'v': v_best,
            'a': a,
            'grid_steps': steps
            }
        return self.image, mask, parameters
    def ramp_correction(self,mask_fit=False,mask=None,model='linear',
                        alpha=1,verbose=False, **kwargs):
        """
        Corrects for phase ramps that appear in ptychography reconstructions. The image need to be phase unwrapped first. 

        Parameters
        ----------
        mask_fit : bool, optional
            If True, a window will pop up to draw a mask around the object. 
            The default is False.
        mask : np.ndarray, optional
            Mask to use for on image that fits values in the mask to a polynomial. 
            The default is None.
        model : str, optional
            Model to use for fitting the masked region. Options are 'linear',
            'quad', and 'cubic'. Recommend 'linear' since it will not change 
            the physical properties of the image. The default is 'linear'.
        alpha : float, optional
            How strongly to apply the fitted polynomial from the image. The default is 1.
        verbose : bool, optional
            Prints the fit result and plots the image of the fit. The default is False.
        **kwargs : list
            Values for coefficients depending on model used. E.g. 'linear' is a*x+b*y+c 
            where [a,b,c] are the coefficients specified. It must start with 'a' and then in order until 
            there are a sufficient number of coefficients.

        Returns
        -------
        np.ndarray
            Image with poly fit subtracted.
        np.ndarray or None
            Mask used for the polynomial fit.
        list
            Fitted coefficients from polynomial fit.

        """
        
        #TODO: the cubic fit is very unstable and often does not necessarily describe anything physical. Use a different fitting model that may be more accurate. 
        model_dict = {
            'linear': self._linear_ramp_model,
            'quad': self._quadratic_ramp_model,
            'cubic': self._cubic_ramp_model,
            }
        
        if model == 'quad':
            warnings.warn("Subtracting a quadratic phase ramp may change "
                          "the physical properties of the image.")
        elif model == 'cubic':
            warnings.warn("Cubic fits absorb real low-frequency signal")
            
        h, w = self.image.shape
        # x = np.arange(w)
        # y = np.arange(h)
        # X, Y = np.meshgrid(x, y)
        x = np.linspace(-0.5,0.5,w)
        y = np.linspace(-0.5,0.5,h)
        X, Y = np.meshgrid(x, y)
        if mask_fit:
            if mask is None:
                self._draw_polygon()
                mask = self._polygon_mask()
            
            x_fit = X[mask].astype(float).ravel() 
            y_fit = Y[mask].astype(float).ravel()
            z_fit = self.image[mask].astype(float).ravel() 
            
            ramp_fit = Model(model_dict[model],independent_vars=['x','y'])
            sig = inspect.signature(model_dict[model])
            param_names = [p for p in sig.parameters if p not in ('x', 'y')]

            # Build params automatically, all set to 0
            params = ramp_fit.make_params(**{name: 0 for name in param_names})
            result = ramp_fit.fit(z_fit,params,x=x_fit,y=y_fit)
            ramp = model_dict[model](X,Y, **result.best_values)
            if verbose:
                print(result.fit_report()) 
                plt.figure(), plt.imshow(ramp,cmap='seismic'), plt.colorbar()
            ramp = model_dict[model](X,Y, **result.best_values)
            return self.image-alpha*ramp, mask, result.best_values 
        else:
            func = model_dict[model]
        
            # Get all parameter names except x,y
            sig = inspect.signature(func)
            param_names = [p for p in sig.parameters if p not in ('x', 'y')]
        
            # Build parameter dict from kwargs
            # Default to 0 if not supplied by user
            user_params = {
                name: kwargs.get(name, 0)
                for name in param_names
            }
        
            # Compute the ramp
            ramp = func(X, Y, **user_params)
        
            return self.image - alpha * ramp, None, user_params
    def _draw_polygon(self, window_name="Draw Polygon"):
        import cv2
        """Draw polygon using mouse clicks, press 'Enter' to finish."""
        if self.image_phase is not None:
            clone = self.image_phase.copy()
        else:
            clone = self.image.copy()
        clone = cv2.normalize(clone, None, 0, 255,cv2.NORM_MINMAX).astype(np.uint8)
        self.mask_points = []
        
        h, w = clone.shape[:2]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h)
        
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mask_points.append((x, y))
                cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow(window_name, clone)

        cv2.imshow(window_name, clone)
        cv2.setMouseCallback(window_name, click_event)

        print("Click to add polygon vertices. Press 'Enter' when done.")
        while True:
            key = cv2.waitKey(1) & 0xFF
            # Enter key finishes
            if key == 13:  # Enter
                break

        cv2.destroyWindow(window_name)

    def _polygon_mask(self):
        import cv2
        """Return boolean mask from drawn polygon."""
        if len(self.mask_points) < 3:
            raise ValueError("At least 3 points required for polygon.")
        
        # Use uint8 for OpenCV
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        pts = np.array(self.mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)  # Fill with 1
    
        # Convert to boolean mask
        mask_bool = mask.astype(bool)
        return mask_bool
    def _linear_ramp_model(self,x, y,a,b,c):
        linear_ramp = a*x+ b*y + c
        return linear_ramp
    def _quadratic_ramp_model(self,x,y,a,b,c,d,e,f):
        quadratic_ramp_model = a*x**2+b*y**2+c*x*y+d*x+e*y+f 
        return quadratic_ramp_model
    def _cubic_ramp_model(self,x,y,a,b,c,d,e,f,g,h,i,j):
        return (
            a*x**3 + b*y**3 + c*x**2*y + d*x*y**2 + 
            e*x**2 + f*y**2 + g*x*y + 
            h*x + i*y +
            j)
    
    def _halo_lowpass(self, phi, method, a, pixel_size=1.0, order=2,
                  frac_alpha=1.0, pad=2):
        """Low-pass phi with the kernel named by `method`, length scale `a`."""
        ny, nx = phi.shape
        py, px = int(pad * ny), int(pad * nx)
        buf = np.zeros((py, px), dtype=float)
        y0, x0 = (py - ny) // 2, (px - nx) // 2
        buf[y0:y0 + ny, x0:x0 + nx] = phi
     
        ky = 2 * np.pi * np.fft.fftfreq(py, d=pixel_size)
        kx = 2 * np.pi * np.fft.rfftfreq(px, d=pixel_size)
        k = np.hypot(ky[:, None], kx[None, :])
        kc = 1.0 / a
     
        if method == 'gaussian':
            G = np.exp(-0.5 * (a * k) ** 2)
        elif method == 'exponential':
            G = (1.0 + (a * k) ** 2) ** -1.5
        elif method == 'cauchy':
            G = np.exp(-a * k)
        elif method == 'butterworth':
            G = 1.0 / (1.0 + (k / kc) ** order)
        elif method == 'fractional':
            G = kc ** frac_alpha / (k ** frac_alpha + kc ** frac_alpha)
        else:
            raise ValueError(f"unknown method: {method!r}")
     
        lp = np.fft.irfft2(np.fft.rfft2(buf) * G, s=(py, px))
        return lp[y0:y0 + ny, x0:x0 + nx]
   
    def halo_correction(self, mask=None, method='cauchy', a_range=None, n_a=40,
                        n_iter=3, alpha=1.0, pixel_size=1.0, order=2,
                        frac_alpha=1.0, verbose=False):
        """
        Correct the nonlinear halo ('glow') artifact by fitting a physical model.
     
        Unlike a polynomial ramp, the halo is object-dependent and localised:
        each feature carries its own skirt. It is modelled as the missing
        low-frequency band of the reconstruction,
     
            phi_meas = phi_true - s * (phi_true (*) g_a)
     
        where g_a is a normalised radial kernel of scale `a`. Inside the vacuum
        mask phi_true == 0, so everything measured there is pure artifact, which
        makes (s, a) a well-posed two-parameter fit.
     
        The residual is LINEAR in s and only nonlinear in a, so `a` is scanned on
        a grid and `s` solved in closed form at each node (profile likelihood).
        This cannot land in a local minimum, unlike a joint nonlinear optimiser.
     
        Parameters
        ----------
        mask : np.ndarray of bool, optional
            True in known-vacuum / flat regions -- same convention as
            ramp_correction_MG. If None, prompts you to draw a polygon.
        method : str
            Kernel: 'cauchy' (default, H ~ a*k at low k -- gradients kept, DC
            lost, the long curved skirt seen experimentally), 'gaussian',
            'exponential', 'butterworth', 'fractional'.
        a_range : (a_min, a_max), optional
            Halo length scale search range, in pixels (or physical units if
            pixel_size is set). Defaults to (2 px, quarter of the short axis).
        n_a : int
            Number of log-spaced nodes in the `a` scan.
        n_iter : int
            Fixed-point iterations. The fit needs an estimate of phi_true; it
            starts from the measured phase and re-estimates. 2-3 is plenty.
        alpha : float
            How strongly to apply the correction, matching ramp_correction.
            alpha=1 fully subtracts the fitted halo.
        verbose : bool
            Plot the fitted halo and the chi2(a) curve.
     
        Returns
        -------
        image : np.ndarray
            Corrected image, same dtype convention as the input (complex in,
            complex out; real phase in, real phase out).
        mask : np.ndarray
            The vacuum mask used.
        parameters : dict
            {'a', 's', 'chi2', 'method', 'a_grid', 'chi2_grid'}
        """
        import warnings
        warnings.warn("Attempting to correct the halo artifact post ptychography "
                      "will not work well. The artifact is not global but localized "
                      "and thus is very difficult to model. This should only be used "
                      "for educational purposes to show the ill-posed problem. "
                      "Removing post ptycho will effect the physical properties of a sample!")
        
        if mask is None:
            self._draw_polygon()
            mask = self._polygon_mask()
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("vacuum mask is empty")
     
        is_complex = np.iscomplexobj(self.image)
        phi = np.angle(self.image) if is_complex else np.asarray(self.image, float)
     
        if a_range is None:
            a_range = (2.0 * pixel_size,
                       0.25 * min(phi.shape) * pixel_size)
        a_grid = np.logspace(np.log10(a_range[0]), np.log10(a_range[1]), n_a)
     
        phi_est = phi.copy()
        a_best = s_best = chi2_best = None
        halo = np.zeros_like(phi)
     
        for _ in range(n_iter):
            chi2_grid = np.empty(n_a)
            s_grid = np.empty(n_a)
            lp_cache = {}
     
            for idx, a in enumerate(a_grid):
                lp = self._halo_lowpass(phi_est, method, a, pixel_size,
                                        order, frac_alpha)
                lp_cache[idx] = lp
                # model in vacuum:  phi_meas ~ s * (-lp)
                A = -lp[mask].ravel()
                b = phi[mask].ravel()
                denom = A @ A
                if denom <= 0:
                    chi2_grid[idx], s_grid[idx] = np.inf, 0.0
                    continue
                s = (A @ b) / denom
                s_grid[idx] = s
                chi2_grid[idx] = np.sum((b - s * A) ** 2)
     
            idx_best = int(np.argmin(chi2_grid))
            a_best = a_grid[idx_best]
            s_best = s_grid[idx_best]
            chi2_best = chi2_grid[idx_best]
     
            halo = -s_best * lp_cache[idx_best]
            phi_est = phi - halo                      # re-estimate phi_true
     
            if a_best in (a_grid[0], a_grid[-1]):
                import warnings
                warnings.warn(
                    f"fitted a={a_best:.3g} sits on the edge of a_range "
                    f"{a_range}; widen the search range.")
     
        if verbose:
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
            im0 = ax[0].imshow(halo, cmap='seismic',
                               vmin=-np.abs(halo).max(), vmax=np.abs(halo).max())
            ax[0].set_title(f"fitted halo  (a={a_best:.2f}, s={s_best:.3f})")
            plt.colorbar(im0, ax=ax[0])
            im1 = ax[1].imshow(phi - alpha * halo, cmap='seismic')
            ax[1].set_title("corrected phase")
            plt.colorbar(im1, ax=ax[1])
            ax[2].semilogx(a_grid, chi2_grid)
            ax[2].axvline(a_best, ls='--', c='k')
            ax[2].set_xlabel("a"), ax[2].set_ylabel(r"$\chi^2$")
            plt.tight_layout(), plt.show(block=False)
     
        phi_out = phi - alpha * halo
        self.image = np.abs(self.image) * np.exp(1j * phi_out) if is_complex \
            else phi_out
     
        parameters = {'a': a_best, 's': s_best, 'chi2': chi2_best,
                      'method': method, 'a_grid': a_grid, 'chi2_grid': chi2_grid}
        return self.image, mask, parameters



def batch_index_generator(N, batch_size):
    """
    Generator that yields index ranges for batching along the first axis.

    Parameters:
        N (int): The first dimension of the dataset (number of images).
        batch_size (int): Number of images per batch.

    Yields:
        tuple: (start_index, end_index) for each batch.
    """
    for i in range(0, N, batch_size):
        yield i, min(i + batch_size, N)    

def nearest_neighbor_order(X, Y):
    # Combine X and Y into a 2D array of points
    points = np.column_stack((X, Y))
    N = len(points)
    
    # Initialize the list of ordered indices with the first point
    ordered_indices = [0]
    remaining_indices = set(range(1, N))  # Start with all indices except the first one
    
    while remaining_indices:
        last_index = ordered_indices[-1]
        last_point = points[last_index]
        
        # Find the nearest neighbor among the remaining points
        distances = [np.linalg.norm(points[i] - last_point) for i in remaining_indices]
        nearest_index = list(remaining_indices)[np.argmin(distances)]
        
        # Append the closest point index to ordered_indices and remove it from remaining_indices
        ordered_indices.append(nearest_index)
        remaining_indices.remove(nearest_index)
    
    return ordered_indices

def subpixel_image_registration(f,peak, window=1.5, upsample=100,n_points=None):
    """
    Discrete Fourier transform via matrix muliplication for subpixel image
    registration. This allows for DFTs for part of an array instead of upsampling the whole image. 
    
    Guizar-Sicairos, ManuelThurman, and J. R. Fienup, “Efficient subpixel image registration algorithms,” 
    Optics Letters, Vol. 33, Issue 2, pp. 156-158, vol. 33, no. 2, pp. 156–158, Jan. 2008, 
    doi: 10.1364/OL.33.000156.
  

    Parameters
    ----------
    f : numpy.ndarray
        Image for subpixel registration. Can be a cross-correlation between two images, masked image of 
        a flat region with linear phase ramp, etc. 
    peak : tuple
        Estimated high pixel. Suggest from 2x sampled R = FFT[f].
    window : float, optional
        Window size for around max value for FFT[f] for subpixel registration. The default is 1.5.
    upsample : int, optional
        How much to upsample window for subpixel registration. The default is 1.

    Returns
    -------
    R : numpy.ndarray
        DFT by matrix multiplacion of f in subpixel mapping around peak
    U : numpy.ndarray
        2D grid of u values from meshgrid.
    V : numpy.ndarray
        2D grid of v values from meshgrid.
    X : numpy.ndarray
        2D grid of x values from meshgrid.
    Y : numpy.ndarray
        2D grid of y values from meshgrid.

    """
    M, N = f.shape 
    u0, v0 = peak 
    
    if n_points is None:
        n_points = int(np.ceil(window*upsample))
    n_points = max(int(n_points),3)|1
    
    x = np.arange(M)
    y = np.arange(N)
    u = np.linspace(u0-window/2,u0+window/2,n_points)
    v = np.linspace(v0-window/2,v0+window/2,n_points)
    Ex = np.exp(-1j*2*np.pi/M*np.outer(u,x))
    Ey = np.exp(-1j*2*np.pi/N*np.outer(v,y)) 
    R = ( Ex @ f ) @ Ey.T
    U, V = np.meshgrid(u,v, indexing='ij') 
    X, Y = np.meshgrid(x,y, indexing='ij')
    return R, U, V, X, Y


def phase_ramp(u,v,dx,dy):
    ramp = xp.exp(-1j*2*xp.pi*(u*dx+v*dy))
    return ramp[np.newaxis,:,:]

def get_RAM_memory():
    memory = psutil.virtual_memory()
    free_mem, total_mem = memory.available, memory.total
    print(f"Free RAM memory: {free_mem / 1e6:.2f} MB")
    print(f"Total RAM memory: {total_mem / 1e6:.2f} MB")

def get_GPU_memory():
    device = cp.cuda.Device(0)  # Select GPU 0 (change if needed)
    free_mem, total_mem = device.mem_info
    print(f"Free memory: {free_mem / 1e6:.2f} MB")
    print(f"Total memory: {total_mem / 1e6:.2f} MB")
    
def clear_GPU_memory():
    cp.cuda.Device(0).synchronize()
    cp.fft.config.get_plan_cache().clear()
    gc.collect() 
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

if __name__ == '__main__':
    None 
   
    
    
    
    
    