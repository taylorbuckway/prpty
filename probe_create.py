# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 08:23:04 2023

@author: tbuckway
"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
#from ptycho_random import numpy_complex
from scipy.special import hermite, genlaguerre
from scipy.ndimage import shift, gaussian_filter
# from zernike import Zernike_polynomial 
# import plot as plot
class probes:
    """Create your own probes"""
    def __init__(self, lam, pixel_size, pixel_number):
        """
        Probe class is used to generate a probe. This uses Gaussian beam equation 
        as described in chapter 11 of "Physics of Light and Optics" by Justin
        Peatross and Micheal Ware

        Parameters
        ----------
        lam : float
            The wavelength of the light.
        pixel_size : float
            Size of the pixel. For CDI, this should correspond to the
            pixels size in real space. 
        pixel_number : int
            The number of pixels. This will be a square image. 

        Returns
        -------
        None.

        """
        self.lam = lam 
        self.pixel_size = pixel_size
        self.pixel_number = pixel_number 
        self.domain_length = self.pixel_size*self.pixel_number 
        self.x = np.linspace(-self.domain_length/2, self.domain_length/2,self.pixel_number)
        self.y = self.x
        self.X, self.Y = np.meshgrid(self.x,self.y) #mapped x and y into a 2d array X and Y. 
    def probe_singleaxis(self, wo, z,rho, Eo=1):
        """
        Single 

        Parameters
        ----------
        wo : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        rho : TYPE
            DESCRIPTION.
        Eo : TYPE, optional
            DESCRIPTION. The default is 1.

        Raises
        ------
        f
            DESCRIPTION.

        Returns
        -------
        E : TYPE
            DESCRIPTION.
        w : TYPE
            DESCRIPTION.

        """
        if z == 0:
            raise f'{z} is undefined for guassian beams equations. add a very small value instead'
        k = 2*np.pi/self.lam
        zo = k*wo**2/2
        R = z + zo**2/z 
        w = wo*np.sqrt(1+z**2/zo**2)
        E = Eo*wo/w*np.exp(-(rho)/w**2)*np.exp(1j*k*z+1j*k*(rho)/2/R-1j*np.arctan(z/zo))
        # E[np.abs(E)<Eo*1/np.exp(1)**6] = 0+1j*0
        return E, w
    
    def Hermite_Gaussian_beam(self,l,m,wo,z,Eo=1,w=None,xo=0,yo=0):
        k = 2*np.pi/self.lam
        zo = k*wo**2/2
        if w != None:
            z = zo*np.sqrt(w**2/wo**2-1)
            print(f'Distance from focus is {z*10**3:.2f} mm')
        else:
            w = wo*np.sqrt(1+(z/zo)**2)
        R = z+zo**2/z 
        N = l+m
        psi = (N+1)*np.arctan(z/zo)
        hermite_probe = Eo*wo/w*hermite(l,monic=True)(np.sqrt(2)*(self.X-xo)/w)*hermite(m,monic=True)(np.sqrt(2)*(self.Y-yo)/w)*np.exp(
            -((self.X-xo)**2+(self.Y-yo)**2)/w**2)*np.exp(-1j*k*((self.X-xo)**2+(self.Y-yo)**2)/(2*R))*np.exp(1j*psi)*np.exp(-1j*k)
        hermite_probe /= np.max(np.abs(hermite_probe))
        hermite_probe *= Eo 
        return hermite_probe
    
    def Laguerre_Gaussian_beam(self,l,m,wo,z,Eo=1,w=None,xo=0,yo=0):
        "m is the radial parameter l is azimuthal and can be positive or negative"
        r = np.sqrt(self.X**2+self.Y**2)
        phi = np.arctan(self.Y/self.X)
        k = 2*np.pi/self.lam
        zo = k*wo**2/2 
        if w != None:
            z = zo*np.sqrt(w**2/wo**2-1)
            print(f'Distance from focus is {z*10**3:.2f} mm')
        else:
            w = wo*np.sqrt(1+(z/zo)**2)
        R = z+zo**2/z
        N = np.abs(l)+2*m
        psi = (N+1)*np.arctan(z/zo)
        C = np.sqrt(2*factorial(m)/(np.pi*factorial(m+np.abs(l))))
        laguerre_probe = C/w*(r*np.sqrt(2)/w)**(np.abs(l))*np.exp(-r**2/w**2)*genlaguerre(m,np.abs(l),monic=True)(2*r**2/w**2)*np.exp(-1j*k*r**2/(2*R))*np.exp(-1j*l*phi)*np.exp(1j*psi)
        laguerre_probe /= np.max(np.abs(laguerre_probe))
        laguerre_probe *= Eo 
        return laguerre_probe
    
    def generate_shared_probes(self,num_shared_probes,wo,z,Etotal,Eweights,w=None,basis='Hermite'):
        modes= {
            'Hermite': self.Hermite_Gaussian_beam,
            'Laguerre': self.Laguerre_Gaussian_beam,
            }
        self.probe = np.ones((1,1,num_shared_probes,self.pixel_number,self.pixel_number),np.complex64)
        Eo = []
        for weight in Eweights:
            Eo.append(Etotal*weight)
        l = 0
        m = 0
        n = 0
        for i in range(num_shared_probes):
            self.probe[0,0,i,:,:] = modes[basis](l, m, wo, z, Eo[i], w)
            if l == n and m == n:
                l+=1
                m = 0
                n+=1
            elif l>m:
                l,m = np.copy(m),np.copy(l)
            elif l<m:
                l+=1
                l,m = np.copy(m),np.copy(l)                    
        return self.probe
    
    def gaussian_probe(self, wo, z, w=None, focrel=1,Eo=1):
        """
        

        Parameters
        ----------
        wo : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        w : float, optional
            Desired beam size. The default is None.
        focrel : float, optional
            Relative to the focus before focus is -1. The default is 1 meaning after focus.

        Returns
        -------
        probe : TYPE
            DESCRIPTION.
        w : TYPE
            DESCRIPTION.

        """
        if w != None:
            zo = np.pi*wo**2/self.lam
            z = focrel*zo*np.sqrt(w**2/wo**2-1)
        self.probe, w = self.probe_singleaxis(wo,z,self.X**2+self.Y**2,Eo=Eo)
        return self.probe, w
    def astigmatic_probe(self, wo, z, ratio=1, focrel = 1, w=None,Eo=1):
        if w != None:
            zo = np.pi*wo**2/self.lam
            z = focrel*zo*np.sqrt(w**2/wo**2-1)
            print(z)
        self.probe, w = self.probe_singleaxis(wo,z,(self.X/ratio)**2+(self.Y)**2,Eo=Eo)
        return self.probe, w/ratio
    def aperture(self, diameter,offset=[0,0],Eo=1):
        self.aper = np.zeros_like(self.X)
        self.aper[(self.X-offset[0])**2+(self.Y-offset[1])**2 <= (diameter/2)**2] = Eo
        return
    def circular_beam_block(self, diameter, offset=[0,0],Eo=1):
        self.beam_block = np.zeros_like(self.X)
        self.beam_block[(self.X-offset[0])**2+(self.Y-offset[1])**2 >= (diameter/2)**2] = Eo 
        return
    def donut_mask(self, innerdi, outerdi,offset=[0,0],Eo=1):
        if outerdi < innerdi:
            raise Exception("The outer diameter needs to be larger than the inner diameter")
        self.aperture(outerdi,offset=offset,Eo=Eo)
        self.circular_beam_block(innerdi,offset=offset,Eo=Eo)
        self.donut = self.aper*self.beam_block
        self.donut = gaussian_filter(self.donut, 2.0)
        return   
    
    def simulate_varying_gaussian_probe(self,num_positions,
                               max_pos_offset,
                               max_z_offset,
                               percent_intensity_offset,
                               wo,
                               z,
                               w=None,
                               rand_techn='normal',
                               Eo=1,
                               verbose=False,
                               ):
        #TODO: include option for different hermite gaussian mode
        self.varying_probe = np.zeros((num_positions,self.pixel_number,self.pixel_number)).astype(np.complex128)
        if rand_techn == 'normal':
            rand_x_offset = np.random.normal(0.0,max_pos_offset,num_positions)
            rand_y_offset = np.random.normal(0.0,max_pos_offset,num_positions)
            rand_z_offset = np.random.normal(0.0,max_z_offset,num_positions)
            rand_Eo = np.random.normal(Eo,Eo*percent_intensity_offset,num_positions)
        elif rand_techn == 'uniform':
            rand_x_offset = np.random.uniform(-max_pos_offset,max_pos_offset,num_positions)
            rand_y_offset = np.random.uniform(-max_pos_offset,max_pos_offset,num_positions)
            rand_z_offset = np.random.uniform(-max_z_offset,max_z_offset,num_positions)
            rand_Eo = np.random.uniform(-Eo*(1-percent_intensity_offset),Eo*(1-percent_intensity_offset),num_positions)
        rand_Eo = np.abs(rand_Eo) #ensures the electric field should be positive
        if w:
            zo = np.pi*wo**2/self.lam
            z = zo*np.sqrt(w**2/wo**2-1)
        for i in range(num_positions):
            self.varying_probe[i] ,_ = self.probe_singleaxis(wo,z+rand_z_offset[i],(self.X-rand_x_offset[i])**2+(self.Y-rand_y_offset[i])**2,Eo=rand_Eo[i])
        if verbose == True:
            fig, ([[ax1,ax2],[ax3,ax4]]) = plt.subplots(2,2)
            ax1.plot(rand_x_offset)
            ax1.set_title('x variation')
            ax2.plot(rand_y_offset)
            ax2.set_title('y variation')
            ax3.plot(rand_z_offset)
            ax3.set_title('z variation')
            ax4.plot(rand_Eo)
            ax4.set_title('Eo variation')
            plt.tight_layout()
        return 
    def simulate_varying_probe(self,
                               probe, 
                               num_positions,
                               max_pos_offset, #in subpixels
                               max_z_offset,
                               # percent_intensity_offset,
                               rand_techn='normal',
                               verbose=False,
                               ):
        self.varying_probe = np.zeros((num_positions,self.pixel_number,self.pixel_number)).astype(np.complex128)
        if rand_techn == 'normal':
            rand_x_offset = np.random.normal(0.0,max_pos_offset,num_positions)
            rand_y_offset = np.random.normal(0.0,max_pos_offset,num_positions)
            rand_z_offset = np.random.normal(0.0,max_z_offset,num_positions)
            # rand_Eo = np.random.normal(Eo,Eo*percent_intensity_offset,num_positions)
        elif rand_techn == 'uniform':
            rand_x_offset = np.random.uniform(-max_pos_offset,max_pos_offset,num_positions)
            rand_y_offset = np.random.uniform(-max_pos_offset,max_pos_offset,num_positions)
            rand_z_offset = np.random.uniform(-max_z_offset,max_z_offset,num_positions)
            # rand_Eo = np.random.uniform(-Eo*(1-percent_intensity_offset),Eo*(1-percent_intensity_offset),num_positions)
        # rand_Eo = np.abs(rand_Eo) #ensures the electric field should be positive
        for i in range(num_positions):
            self.varying_probe[i] = self.shift_complex_image_subpixel(probe,rand_x_offset[i],rand_y_offset[i])
            self.varying_probe[i] = self.z_shift(self.varying_probe[i], rand_z_offset[i])
        if verbose == True:
            fig, ([[ax1,ax2],[ax3,ax4]]) = plt.subplots(2,2)
            ax1.plot(rand_x_offset)
            ax1.set_title('x variation')
            ax2.plot(rand_y_offset)
            ax2.set_title('y variation')
            ax3.plot(rand_z_offset)
            ax3.set_title('z variation')
            # ax4.plot(rand_Eo)
            # ax4.set_title('Eo variation')
            # plt.tight_layout()
        return
    def shift_complex_image_subpixel(self,image, shift_x=0.0, shift_y=0.0):
        # Use scipy.ndimage.shift to apply subpixel shifts
        shifted_image = shift(image, shift=(shift_y, shift_x), order=3)  # Order=3 for cubic interpolation
        return shifted_image
    def z_shift(self,image,shift_z):
        return np.abs(image)*np.exp(1j*np.angle(image)+1j*2*np.pi*shift_z/self.lam)

    def simulate_varying_probe_tike(
            self,
            scan, 
            eigen_probe_modes,
            wo,
            z,
            probe=np.array([]),
            Eo=1,
            w=None,
            probe_variation=None, # first is for the zeroth order 0.2 is the other.
            wmethod='sinusoidal',
            offsets = None,
            period = None,
            verbose=False):
        from tike.ptycho.probe import add_modes_cartesian_hermite
        from tike.ptycho.probe import get_varying_probe
        from tike.precision import floating
        weight_method = {
            'sinusoidal': self.simulate_varying_weights_sinusoidal,
            'random': self.simulate_varying_weights_random,
            'normal': self.simulate_varying_weights_normal
            }
        if probe_variation == None:
            self.probe_variation = np.ones((1,eigen_probe_modes,1))
        else:
            if len(probe_variation) != eigen_probe_modes:
                num_added = eigen_probe_modes-len(probe_variation)
                last_element = probe_variation[-1]
                probe_variation.extend([last_element]*num_added)
            self.probe_variation = np.array(probe_variation).reshape((1,eigen_probe_modes,1))
        if len(probe.shape) == 2:
            eigen_probe = add_modes_cartesian_hermite(np.reshape(probe,(1,1,1,probe.shape[0],probe.shape[1])), eigen_probe_modes).reshape(1,eigen_probe_modes,1,probe.shape[0],probe.shape[1])
        else:
            self.generate_shared_probes(num_shared_probes=eigen_probe_modes, wo=wo, z=z,Eo=Eo,w=w)
            eigen_probe = self.probe.reshape((1,self.probe.shape[2],1,self.probe.shape[-2],self.probe.shape[-1]))
        eigen_weights = weight_method[wmethod](scan, eigen_probe,offsets=offsets,period=period).astype(floating)
        if len(probe.shape) == 2:
            # shared_probe = probe.reshape(1,1,1,probe.shape[-2],probe.shape[-1])
            shared_probe = eigen_probe[:,:1,:,:,:]
        else:
            # shared_probe = self.probe[:,:,0,:,:].reshape(1,1,1,self.probe.shape[-2],self.probe.shape[-1])
            shared_probe = eigen_probe[:,:1,:,:,:]
        eigen_probe = eigen_probe[:,1:,:,:,:]
        # eigen_weights[:,0] = np.random.uniform(self.probe_variation[0],self.probe_variation[0]+0.09,eigen_weights[:,0].shape).astype(floating)
        self.varying_probe = get_varying_probe(shared_probe,eigen_probe,eigen_weights)[:,0,0,:,:]
        if eigen_probe_modes <= 1:
            eigen_probe = None
        if verbose == True:
            try:
                plot.plot_eigen_probes(shared_probe, eigen_probe,method='abs')
                ani = plot.video(np.abs(self.varying_probe),interval=50,repeat_delay=500)
                plt.figure()
                for i in range(eigen_probe_modes):
                    plt.plot(eigen_weights[:,i,0],label=f'Eigen mode: {i}')
                    plt.xlabel('Position number')
                    plt.ylabel('Weight')
                    plt.legend()
            except AttributeError:
                plot.plot_phase(shared_probe[0,0,0])
                ani = plot.video(np.abs(shared_probe[0,0]))
            return shared_probe, eigen_probe, eigen_weights, ani
        else:
            return shared_probe, eigen_probe, eigen_weights
    def simulate_varying_weights_sinusoidal(self,scan, eigen_probe, offsets=None, period=None):
        """Taken and modified from Tike
        Generate weights for eigen probe that follow random sinusoid.
    
        The amplitude of the of weights is 1, the phase shift is (0, 2π], and the
        period is at most one full scan.
        """
        N = scan.shape[0]
        x = np.arange(N)[..., :, None, None]
        if not period:
            period = N * np.random.rand(*eigen_probe.shape[:-2])
        else:
            try:
                period = np.array(period).reshape(*eigen_probe.shape[:-2])
            except:
                print('The number of periods need to match the number of eigen modes')
        phase = 2 * np.pi * np.random.rand(*eigen_probe.shape[:-2])
        print(f'period is {period}')
        if offsets:
            if len(offsets) != eigen_probe.shape[1]:
                num_added = eigen_probe.shape[1]-len(offsets)
                last_element = offsets[-1]
                offsets.extend([last_element]*num_added)
            offsets = np.array(offsets).reshape(*eigen_probe.shape[:-2])
            return self.probe_variation*np.sin(2 * np.pi / period * x - phase)+offsets
        else:
            return self.probe_variation*np.sin(2 * np.pi / period * x - phase)
    def simulate_varying_weights_random(self,scan,eigen_probe, offsets=None, period=None):
        scan_N = scan.shape[0]
        eigen_N = eigen_probe.shape[1]
        num_modes = eigen_probe.shape[2]
        if offsets:
            if len(offsets) != eigen_probe.shape[1]:
                num_added = eigen_probe.shape[1]-len(offsets)
                last_element = offsets[-1]
                offsets.extend([last_element]*num_added)
            offsets = np.array(offsets).reshape(*eigen_probe.shape[:-2])
            return self.probe_variation*np.random.uniform(-1,1,(scan_N,eigen_N,num_modes))+offsets
        else:
            return self.probe_variation*np.random.uniform(-1,1,(scan_N,eigen_N,num_modes))
    def simulate_varying_weights_normal(self,scan,eigen_probe, offsets=None, period=None):
        scan_N = scan.shape[0]
        eigen_N = eigen_probe.shape[1]
        num_modes = eigen_probe.shape[2]
        if offsets:
            if len(offsets) != eigen_probe.shape[1]:
                num_added = eigen_probe.shape[1]-len(offsets)
                last_element = offsets[-1]
                offsets.extend([last_element]*num_added)
            offsets = np.array(offsets).reshape(*eigen_probe.shape[:-2])
            return np.random.normal(0,self.probe_variation,(scan_N,eigen_N,num_modes))+offsets
        else:
            return np.random.normal(0,self.probe_variation,(scan_N,eigen_N,num_modes))
    def simulate_varying_weights_expdecay(self,scan, eigen_probe):
        num_eig = eigen_probe.shape[1]
        N = scan.shape[0]
        centers = np.exp(-1*np.linspace(0.06,2.9,num_eig))
        eigen_weights = np.ones((N,num_eig,1))
        for i in range(num_eig):
            eigen_weights[:,i,0] *= np.random.uniform(centers[i]-0.05,centers[i]+0.05,(N))
        return eigen_weights
        
        
        
if __name__ == '__main__':
    # import fourier_operators as fo
    from scipy.ndimage import center_of_mass, shift
    import plot
    import measurements
    from skimage.util import random_noise 
    import tifffile
    # import fourier_operators as fo
    # import initialize_probe
    wavelength = 30e-9
    energy = 1239.842/(wavelength*10**9)
    binto = 4
    crop = 1024
    pixel_size = 13.5e-6*binto # pixel size of detector
    num_pixels = crop//binto
    distance = 0.14112
    recon_pixel_size = distance*wavelength/pixel_size/num_pixels
    wo = 5.9e-6
    w = 20e-6

    a = 2*w 
    probegen = probes(wavelength,recon_pixel_size,num_pixels)
    probe1 = probegen.Laguerre_Gaussian_beam(-1, 0, 5.0e-6, 0.00001)
    probe2 = probegen.Laguerre_Gaussian_beam(0, 0, 5.0e-6, 0.00001)
    plot.plot_phase(probe1+2*probe2)
