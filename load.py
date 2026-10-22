# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:37:32 2023

@author: tbuckway
"""
import numpy as np
import glob
from natsort import natsorted
import h5py
from scipy.ndimage import center_of_mass
import os


mm = 10**-3 # millimeters
um = 10**-6 # microns

class LoadData:
    def __init__(self,file=None,path=None):
        """
        General load data module. Files should be asc file generated from
        Andor solace software. 

        Parameters
        ----------
        file : string, optional
            File to load. The default is None.
        path : string, optional
            Path the file is located in. The default is None.

        Returns
        -------
        None.

        """
        self.file = file
        self.path = path 
    def LoadSingle(self,skiprows=0,usecols=None):
        """
        Loads a single file.

        Parameters
        ----------
        skiprows : int, optional
            How many rows to skip. The default is 0.
        usecols : int, optional
            Which columns to use. The default is None.

        Returns
        -------
        array
            Data as a numpy array.

        """
        self.data = np.loadtxt(self.file,skiprows=skiprows,usecols=usecols)
        return self.data
    def LoadMany(self, skiprows=0,usecols=None):
        self.filenames = natsorted(glob.glob(self.path))
        self.data = []
        for file in self.filenames:
            self.temp = np.loadtxt(file,skiprows=skiprows,usecols=usecols)
            self.data.append(self.temp)
        self.data = np.array(self.data)
        return self.data
class Loadrand:
    def __init__(self,file=None):
        import scans
        """
        Generates a ptycho dataset with random arrays set for data. For testing
        ptycho code.

        Parameters
        ----------
        file : string,
            Dummy variable does not do anything

        Returns
        -------
        None.

        """
        positions = scans.scan('fermat', (0,0), 15e-6, 15e-6, 1e-6)
        positions[:,0] -= np.min(positions[:,0])
        positions[:,1] -= np.min(positions[:,1])
        data = np.random.randint(300,2**16,size=(positions.shape[0],64,64))
        
        self.data = data
        self.background = np.random.randint(250,350,size=(64,64))
        self.positions = positions
        self.distance = 50e-3
        self.x_pixel_size = 13.5e-6  
        self.y_pixel_size = 13.5e-6
        self.energy = 42 #eV
        self.wavelength = 1239.842/42*10**-9
    
class LoadHHG:
    def __init__(self,file,mode='r',raw_data=False):
        """
        Load HHG data from BYU tabletop EUV source. It takes a hdf5 like file 
        and loads the data and separates it into different variables. It puts the data 
        into a class object. 
        Example of use:
            
        ld = LoadHHG(filename)
        ld.data #this variable returns the data. 

        Parameters
        ----------
        file : string,
            Hdf5 like file containing the data.

        Returns
        -------
        None.

        """
        self.file = file
        self.f = h5py.File(self.file,mode)
    # def ALSChesnel(self):
    #     self.data = self.f['entry0/counter0/data'].value
    #     self.energy = self.f['entry0/counter0/energy'].value
    #     self.polarization = self.f['entry0/motors/POLARIZATION'].value
    # def HHG(self):
        try:
            if raw_data == True:
                self.data = self.f['data/data']
            else:
                self.data = self.f['data/processed_data']
        except KeyError:
            self.data = self.f['data/data']
        self.background = np.array(self.f['data/background'])
        self.positions = np.squeeze(self.f['data/position'])*mm
        self.distance = np.array(self.f['equipment/detector/distance'])*um
        self.x_pixel_size = np.array(self.f['equipment/detector/x_pixel_size'])*um  
        self.y_pixel_size = np.array(self.f['equipment/detector/y_pixel_size'])*um
        self.energy = np.array(self.f['equipment/source/energy']) # eV
        self.wavelength = np.array(self.f['equipment/source/wavelength'])*um
class LoadHeNe:                                      
    def __init__(self,file,mode='r',raw_data=False):
        """
        Load HeNe data from BYU optical setup. It takes a hdf5 like file 
        and loads the data and separates it into different variables. It puts the data 
        into a class object. 
        Example of use:
            
        ld = LoadHeNe(filename,distance=10)
        ld.data #this variable returns the data. 

        Parameters
        ----------
        file : string
            Hdf5 like file ontaining the data. 
        distance : float
            Distance from sample to detector. If using a fourier transforming lens
            then it is the focal length of the lens. 
        energy : flaot, optional
            The photon energy of the HeNe laser. If using a different 
            laser input that photon energy. The default is 1.96.
        x_pixel_size : float, optional
            Pixel size of detector. The default is 5.5*um.

        Returns
        -------
        None.

        """
        self.file = file
        self.f = h5py.File(self.file,mode)
        self.data = self.f['data/rot_0/det_0']
        try: 
            self.true_probe = self.f['data/rot_0/true_probes'][...]
            self.shared_probe = self.f['data/rot_0/shared_probe'][...]
            self.true_psi = self.f['data/rot_0/true_psi'][...]
            self.true_scan = self.f['data/rot_0/true_scan'][...]
            self.eigen_probe = self.f['data/rot_0/eigen_probe'][...]
            self.eigen_weight = self.f['data/rot_0/eigen_weights'][...]
        except KeyError:
            try:
                self.true_probe = self.f['data/rot_0/true_probes'][...]
                self.true_psi = self.f['data/rot_0/true_psi'][...]
                self.true_scan = self.f['data/rot_0/true_scan'][...]
                self.shared_probe = self.f['data/rot_0/shared_probe'][...]
                self.eigen_weight = self.f['data/rot_0/eigen_weights'][...]
            except KeyError:
                None
        try:
            self.probes = self.f['data/rot_0/det_1']
        except KeyError: 
            None
        self.positions = np.array(self.f['data/rot_0/position'])*mm
        self.background = np.array(self.f['equipment/det_0/background'])
        self.distance = self.f['equipment/det_0/distance'][...].tolist()
        self.x_pixel_size = self.f['equipment/det_0/x_pixel_size'][...].tolist()*um
        self.energy = self.f['equipment/source/energy'][...].tolist()
        self.wavelength = 1239.842/self.energy*10**-9
        # self.f.close()
class LoadAPS_2IDD:
    def __init__(self,file,mode='r'):
        """
        Load APS data from Sector ID2. It takes a hdf5 like file 
        and loads the data and separates it into different variables. It puts the data 
        into a class object. 
        Example of use:
            
        ld = LoadAPS_ID2(filename)
        ld.data #this variable returns the data. 

        Parameters
        ----------
        file : string,
            This is actually the folder containing the data and experimental paramters.

        Returns
        -------
        None.

        """
        try:
            # List all entries in the specified directory
            for entry in os.listdir(file):
                # Construct the full path of the entry
                full_path = os.path.join(file, entry)
    
                # Check if the entry is a file and its name matches the phrase exactly (case-sensitive)
                if os.path.isfile(full_path) and '_dp' in entry:
                    self.f = h5py.File(full_path,mode)
                if os.path.isfile(full_path) and '_para' in entry:
                    self.params = h5py.File(full_path,mode)
        except FileNotFoundError:
            raise print(f"Error: Directory not found at '{file}'")
        except Exception as e:
            raise print(f"An error occurred: {e}")
        self.data = self.f['dp']
        self.background = np.zeros(self.data.shape[-2:])
        px = self.params['ppX'][...]
        py = self.params['ppY'][...]
        self.positions = np.stack((py,px),axis=1)
        self.recon_pixel_size = self.params['dx'][...]
        self.x_pixel_size = 75e-6 
        self.y_pixel_size = 75e-6
        self.energy = (1239.842/(self.params['lambda'][...]*10**9))[0] #eV
        self.wavelength = self.params['lambda'][...]
        self.distance = (self.recon_pixel_size*self.x_pixel_size*self.data.shape[-1]/self.wavelength)[0]


class LoadRecon:
    def __init__(self,file):
        """
        Loads a ptychography reconstruction from a hdf5 like file and separates 
        the data into different variables
        Example of usage:
            
        ld = LoadRecon(filename)
        ld.psi #prints the reconstructed object. 

        Parameters
        ----------
        file : string
            The hdf5 like file that contains the reconstruction.

        Returns
        -------
        None.

        """
        self.file = file 
        self.f =  h5py.File(self.file,'r')
        self.probe = self.f['reconstruction/probe'][...]
        self.psi = np.array(self.f['reconstruction/psi'][...])
        self.scan = np.array(self.f['reconstruction/scan'][...])
        self.costs = np.array(self.f['reconstruction/costs'][...])
        self.times = np.array(self.f['reconstruction/times'][...])
        self.pixel_size = np.array(self.f['reconstruction/pixel_size'][...])
        try:
            self.iteration_number = self.f['obj_probe_series/iteration_number'][...]
            self.obj_series = self.f['obj_probe_series/obj_series']
            self.probe_series = self.f['obj_probe_series/probe_series']
            self.eigen_probe = self.f['reconstruction/eigen_probe'][...]
            self.eigen_weight = self.f['reconstruction/eigen_weights'][...]
            self.eigen_probe_series = self.f['obj_probe_series/eigen_probe_series']
            self.eigen_weight_series = self.f['obj_probe_series/eigen_weight_series']
        except KeyError:
            None
        # self.f.close()
class LoadKineticScan:
    def __init__(self,file,imgsize, numimgs,skiprows=34):
        self.file = file
        self.imgsize = imgsize
        self.numimgs = numimgs
        self.data = np.loadtxt(file,skiprows=skiprows)
        self.data = np.reshape(self.data,(self.numimgs,self.imgsize[0],self.imgsize[1]+1))
        self.data = self.data[:,:,1:]
class Load_pty:
    def __init__(self,file,mode='r'):
        """
        General load file for hdf5 like files particularly pty files. 

        Parameters
        ----------
        file : string
            File containing a hdf5 like file.
        mode : string, optional
            How to open the file. The default is 'r' which is read mode.

        Returns
        -------
        None.

        """
        self.file = file 
        self.dataset = h5py.File(self.file,mode)
def extract_from_hdf5(name,h5file):
    with h5py.File(h5file,'r') as hf:
        try:
            extract = hf[name][...]
            return extract 
        except:
            print(f"Acceptable key words are {hf.keys()}")

        
            

if __name__ == '__main__':
    pass    
    
    