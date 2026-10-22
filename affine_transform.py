# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:11:23 2024

@author: tbuckway
"""

import numpy as np
import matplotlib.pyplot as plt
import prpty.plot as plot 

class affine_tranform:
    def __init__(self,scan,verbose=False):
        """
        

        Parameters
        ----------
        scan : numpy.ndarray
            The scan that is to be altered. Units should be pixels.
        verbose : bool, optional
            Plots the original and adjusted scan on same plot. The default is False.

        Returns
        -------
        None.

        """
        # self.scan_aug = np.hstack((scan,np.ones((scan.shape[0],1))))
        self.scan = scan
        self.orig_scan = np.copy(scan)
        self.verbose = verbose
        self.midpoint = (np.max(self.orig_scan,0)+np.min(self.orig_scan,0))//2
    def recenter(self):
        self.scan = self.scan-((np.max(self.scan,0)+np.min(self.scan,0))//2-self.midpoint)
    def change_scan(self,transform):
        scan_aug = np.hstack((self.scan,np.ones((self.scan.shape[0],1))))
        for i in range(self.scan.shape[0]):
            self.scan[i] = np.matmul(transform,scan_aug[i])[:2]
        return
    def translation(self,error):
        etrans = np.random.randint(-error,error,size=self.scan.shape)
        self.scan = self.scan+etrans
        if self.verbose:
            plt.figure()
            plt.title(f'Random positional error up to {error} pixels')
            plot.plot_positions(self.scan,self.orig_scan)
        return
    def scale(self,scalex,scaley):
        Scale = np.array([[scalex,0,0],[0,scaley,0],[0,0,1]])
        self.change_scan(Scale)
        self.recenter()
        if self.verbose:
            plt.figure()
            plt.title(f'Scale error in x: {scalex}, y: {scaley} ')
            plot.plot_positions(self.scan,self.orig_scan)
        return
    def rotate(self,theta):
        theta = theta*np.pi/180
        Rotate = np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
        self.change_scan(Rotate)
        self.recenter()
        if self.verbose:
            plt.figure()
            plt.title(f'Rotation error by {np.round(theta*180/np.pi).astype(int)} degrees')
            plot.plot_positions(self.scan,self.orig_scan)
        return
    def shear(self,shearx,sheary):
        Shear = np.array([[1,shearx,0],[sheary,1,0],[0,0,1]])
        self.change_scan(Shear)
        self.recenter()
        if self.verbose:
            plt.figure()
            plt.title(f'Shear error in x: {shearx} and y: {sheary}')
            plot.plot_positions(self.scan,self.orig_scan)
        return 
        


if __name__ == '__main__':
    import scan
    
    pos, _ = scan.xy_scan('fermat', (0.0,0.0), 20, 20, 1,random=False)
    pos[:,0] -= np.min(pos[:,0])
    pos[:,1] -= np.min(pos[:,1])
    pos = np.round(pos//.150).astype(int)
    print(_)
    positions = affine_tranform(pos,True)
    positions.shear(0.0,0.0)
    
    