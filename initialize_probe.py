# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:17:41 2024

@author: taylo
"""

# try:
#     from tike.ptycho.probe import add_modes_cartesian_hermite, init_varying_probe
# except ImportError:
#     None
import numpy as np
# from sklearn.utils.extmath import randomized_svd
from prpty.fourier_operators import prop_fft, inverse_prop_fft, fresnel_prop
from scipy.ndimage import gaussian_filter, median_filter
# from scipy.linalg import svd 
# from tike.linalg import cov

# class initialize_probe:
#     def __init__(self,probes,num_shared,num_eigen,modes_vary=1):
#         self.probes = probes
#         self.num_shared = num_shared
#         self.num_eigen = num_eigen
#         # self.probes = add_modes_cartesian_hermite(self.probes, num_shared)
#         # shared_probe, eigen_probe, eigen_weights = self.create_shared_and_eigen_modes(num_modes=modes_vary)
#         self.create_shared_and_eigen_modes(num_modes=modes_vary)
#         # return shared_probe, eigen_probe, eigen_weights
#         return 
#     def run_SVD(self,probes):
#         transprobe = np.transpose(probes,(1,2,0))
#         flattened_probes = transprobe.reshape(-1,transprobe.shape[-1])
#         U, S, Vt = randomized_svd(flattened_probes, self.num_eigen,n_oversamples=10,n_iter='auto',power_iteration_normalizer='auto')
#         unflattened_U = U.reshape((probes.shape[1],probes.shape[2],self.num_eigen))
#         U_transposed = np.transpose(unflattened_U,(2,0,1))
#         return U_transposed, S, Vt
#     def create_shared_and_eigen_modes(self,num_modes=1):
#         U,S,Vt = self.run_SVD(self.probes)
#         self.S = S
#         self.shared_probe = add_modes_cartesian_hermite((U[0]).reshape(1,1,1,U[0].shape[0],U[0].shape[1]),self.num_shared)
#         self.shared_probe *= S[0]
#         shared_probes = add_modes_cartesian_hermite(self.probes.reshape(self.probes.shape[0],1,1,self.probes.shape[1],self.probes.shape[2]),self.num_shared)
#         if num_modes == 1:
#                 self.eigen_probes = np.reshape(U[1:]*S[1:].reshape((U[1:].shape[0],1,1)),(1,self.num_eigen-1,num_modes,self.probes.shape[1],self.probes.shape[2]))
#                 self.eigen_weights = np.transpose(Vt).reshape(Vt.shape[1],S.shape[0],num_modes)
#         else:
#             eigen_probes = np.zeros((1,self.num_eigen-1,num_modes,self.probes.shape[1],self.probes.shape[2]))
#             eigen_weights = np.zeros((self.probes.shape[0],S.shape[0],num_modes))
#             for i in range(num_modes):
#                 if i == 0:
#                     eigen_probes[0,:,i,:,:] = U[1:]*S[1:].reshape((U[1:].shape[0],1,1))
#                     eigen_weights[:,:,i] = Vt
#                 else:
#                     U, S, Vt = self.run_SVD(shared_probes[:,0,i])
#                     eigen_probes[0,:,i,:,:] = U[1:]*S[1:].reshape((U[1:].shape[0],1,1))
#                     eigen_weights[:,:,i] = Vt
#         return
    
# class initalize_eigen_probe:
#     def __init__(self,probes,num_eigen,data,lossmodel='Amplitude'):
#         model = {
#             'Poisson': self.Poisson_log_likelihood(),
#             'Amplitude': self.Amplitude_log_likelihood()
#             }
#         self.data = data 
#         self.num_positions = probes.shape[0]
#         self.probe_shape = probes.shape[-1] # assuming square images
#         self.shared_probe = np.average(probes,0).reshape((1,1,1,self.probe_shape,self.probe_shape))
#         self.num_eigen = num_eigen
#         self.varying_probe = probes.astype(np.complex128)
#         self.lossmodel = lossmodel
#         scan = np.random.random((self.num_positions,2))
#         self.eigen_probe, self.eigen_values = init_varying_probe(scan=scan, shared_probe=self.shared_probe, num_eigen_probes=self.num_eigen)
#         self.forward_prop()
#         #TODO: put in a more accurate version for A see odstricl's paper
#         self.A = np.diag(np.ones(self.probe_shape))
#         Loss = model[lossmodel]
#         return
#     def Poisson_log_likelihood(self):
#         return -1*np.sum(self.data*np.log(self.e_data)-self.e_data,axis=(-1,-2))
#     def Amplitude_log_likelihood(self):
#         sigma_e = np.zeros_like(self.data)
#         self.sigma_a = 0.5*np.sqrt(1+sigma_e**2/self.data)
#         return np.sum(1/2*(((np.sqrt(self.data)-np.sqrt(self.e_data))/self.sigma_a))**2,axis=(-1,-2))
#     def forward_prop(self):
#         obj = (np.ones((self.probe_shape,self.probe_shape))*0.5).astype(np.complex128)
#         self.Psi = np.zeros_like(self.data)
#         for position in range(self.num_positions):
#             self.Psi[position] = prop_fft(self.varying_probe[position]*obj,center=False)
#         self.data_e = np.abs(self.Psi)**2
#         return
#     def modulus_constraint(self,x):
#         return 
#     def grad_prop_exitwave(self):
#         if self.lossmodel == 'Amplitude':
#             W = 1/(2*self.sigma_a)**2 
#             self.grad = 2*W*np.sum(self.A*(1-np.sqrt(self.data/self.data_e))*self.Psi,axis=(-1,-2))
#         else:
#             self.grad = np.sum((self.A*(1-self.data/self.data_e)*self.Psi),axis=(-1,-2))
#         return
#     def calc_opt_Psi(self):
#         if self.lossmodel == 'Amplitude':
#             alpha = 1/np.sum(self.A,axis=0)
#         else:
#             #TODO: figure out how to replace 0 with alpha. This is a recursive function
#             alpha = 0
#             xi = 1-self.data/self.data_e
#             alpha = 1/np.sum(self.A,axis=0)*np.sum(self.data_e-xi*(self.data/1-alpha*xi),axis=(-1,-2))/np.sum(xi**2*self.data_e,axis=(-1,-2))          
#         self.opt_Psi = self.Psi - alpha*self.grad
#         return 
    
            
        
        

def add_guessed_phase(
        diff,
        wavelength,
        z,
        det_pixel,
        percentage=0.05,
        sigma = 1.0, 
        threshold=True,
        top_hat=False,
        center_method='CM'
        ):
        avedif = np.average(diff,0)
        if threshold == True:
            avedif[avedif<=percentage*avedif.max()] = 0
            avedif = gaussian_filter(avedif,sigma)
        if top_hat:
            avedif[avedif>1] = 1
        probe = inverse_prop_fft(avedif,center_method=center_method)
        probe = median_filter(np.abs(probe),2)*np.exp(1j*np.angle(probe))
        if z == 0:
            return probe
        probe = fresnel_prop(obj=probe,wavelength=wavelength,z=z,det_pixel=det_pixel)
        return probe
        
        
        

if __name__ == "__main__":
    import load as ld 
    data = ld.extract_from_hdf5('data', 'your/path/name') 
    wavelength = 29e-9
    probe = add_guessed_phase(
        diff=data, 
        wavelength=wavelength,
        z=0.0,
        det_pixel=13.5e-6,
        percentage=0.05,
        sigma=1.0,
        threshold=True,
        top_hat=False,
        center_method='CM')
    import plot 
    plot.plot_phase(probe)
    

