# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:55:02 2024

@author: tbuckway
"""

import numpy as np
from math import factorial as fact


def Zernike_polynomial(n,m,size):
    x = np.linspace(-1,1,size)
    y = x
    X, Y = np.meshgrid(x,y)
    Rho = np.sqrt(X**2+Y**2)
    Theta = np.arctan2(Y,X)
    R = np.zeros_like(Rho,dtype=complex)
    if (n-abs(m))%2 == 0:
        for k in range((n-abs(m))//2+1):
            coef = (-1)**k*fact(n-k)/(fact(k)*fact((n+abs(m))//2-k)*fact((n-abs(m))//2-k))              
            R += coef*Rho**(n-2*k)
    Phi = np.exp(1j*m*Theta)
    Z = R*Phi
    if m >= 0: 
        return np.real(Z)
    elif m < 0: 
        return np.imag(Z)
    else:
        return Z

class varying_probe_zernike:
    def __init__(self,probe,num_eigen_probe,num_position):
        self.probe = probe
        self.shape = probe.shape[-1] #assuming square arrays
        self.eigen_probe = np.random.random((1,num_eigen_probe-1,1,self.shape,self.shape)).astype(np.complex128())
        self.eigen_weights = np.random.random((num_position,num_eigen_probe,1))
        self.num_eigen_probe = num_eigen_probe
        self.num_position = num_position
        self.aber_types = {
            'h_tilt': self.h_tilt(),
            'v_tilt': self.v_tilt(),
            'defocus': self.defocus(),
            'oblique_astig': self.oblique_astig(),
            'vert_astig': self.vert_astig()
            }
    def h_tilt(self):
        zern = Zernike_polynomial(1, 1, self.shape)
        return zern 
    def v_tilt(self):
        zern = Zernike_polynomial(1, -1, self.shape)
        return zern
    def defocus(self):
        zern = Zernike_polynomial(2, 0, self.shape)
        return zern 
    def oblique_astig(self):
        zern = Zernike_polynomial(2, 2, self.shape)
        return zern 
    def vert_astig(self):
        zern = Zernike_polynomial(2, -2, self.shape)
        return zern 
    def generate_eigen_probes(self,specify=(None)):
        if specify is not None:
            for i in range(len(specify)):
                self.eigen_probe[0,i,0,:,:] = self.probe[0,0,0,:,:]*(1+1j)*self.aber_types[specify[i]]
        else:
            aber_list = list(self.aber_types.values())
            for aber in range(self.num_eigen_probe-1):
                self.eigen_probe[0,aber,0,:,:] = self.probe[0,0,0,:,:]*(1+1j)*aber_list[aber]
        return
    def generate_eigen_weights(self,periods=None,amps=None,offsets=None):
        for i in range(self.num_eigen_probe):
            if periods is not None:
                period = periods[i] 
            else:    
                period = np.random.randint(0,self.num_position)
            phase = np.random.uniform(-np.pi,np.pi)
            if amps is not None:
                amp = amps[i]
            else:
                amp = np.random.random()
            if offsets is not None:
                offset = offsets[i]
            else:
                offset = np.random.uniform(-2,2)
            t = np.linspace(0,self.num_position-1,self.num_position)
            weight = amp*np.sin(2*np.pi/period*t+phase)+offset 
            self.eigen_weights[:,i,0] = weight
        return 
    

if __name__ == '__main__':
    import plot
    import probe_create as pc
    import matplotlib.pyplot as plt
    from tike.ptycho.probe import get_varying_probe
    probegen = pc.probes(29e-9, 150e-9,256)
    probe = probegen.Hermite_Gaussian_beam(0,0,2.5e-6,0.001,1000)
    probe = probe.reshape((1,1,1,256,256))
    zern = varying_probe_zernike(probe, 1, 100)
    # zern.generate_eigen_probes(['oblique_astig'])
    zern.generate_eigen_weights(periods=[10,10],offsets=[1.0,0.0],amps=[0.1,5])
    probes = get_varying_probe(probe,zern.eigen_probe,zern.eigen_weights)
    ani = plot.video(np.abs(probes[:,0,0]),interval=100,repeat_delay=500)
    # plot.plot_eigen_probes(probe, zern.eigen_probe,'abs')
    plot.plot_eigen_weights(zern.eigen_weights)
    plt.figure(), plt.plot(np.abs(probes[:,0,0,128,128]))
    
    

                