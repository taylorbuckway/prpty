# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:13:48 2023

@author: tbuckway
"""


import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
from matplotlib import collections
import numpy as np
from scipy.ndimage import shift


def videolog(data,interval=100,repeat_delay=1000,vmin=None,vmax=None):
    figs, ax = plt.subplots()
    ims=[]
    for i in range(len(data)):
        im = ax.imshow(np.log(data[i]),vmin=vmin,vmax=vmax,animated=True)
        if i == 0:
            ax.imshow(np.log(data[i]),vmin=vmin,vmax=vmax)
        ims.append([im])
    ani = animation.ArtistAnimation(figs, ims, interval=interval, repeat_delay=repeat_delay)
    return ani

def video(data,interval=500,repeat_delay=3000,vmin=None,vmax=None,cmap='viridis'):
    figs, ax = plt.subplots()
    
    im = ax.imshow(data[0], animated=True, vmin=vmin, vmax=vmax,cmap=cmap)
    
    # cbar = figs.colorbar(im, ax=ax)
    ax.axis('off')
    ims=[]
    for i in range(len(data)):
        im = ax.imshow(data[i],animated=True,vmin=vmin,vmax=vmax,cmap=cmap)
        # figs.colorbar(im,ax=ax)
        ims.append([im])
    ani = animation.ArtistAnimation(figs, ims, interval=interval, repeat_delay=repeat_delay, blit=True)
    return ani

def plot_phase(image,logscale=False):
    if logscale is True:
        fig, (ax1,ax2) =plt.subplots(1,2)
        im1 = ax1.imshow(np.log(np.abs(image)))
    else:
        fig, (ax1,ax2) =plt.subplots(1,2)
        im1 = ax1.imshow(np.abs(image))
    ax1.set_title('Magnitude')
    plt.colorbar(im1,orientation='horizontal')
    im2 = ax2.imshow(np.angle(image),cmap='seismic')
    ax2.set_title('Phase')
    plt.colorbar(im2,orientation='horizontal')
    return ax1, ax2

def plot_complex(image):
    fig, (ax1,ax2) =plt.subplots(1,2)
    im1 = ax1.imshow(np.real(image),cmap='seismic')
    ax1.set_title('Real')
    plt.colorbar(im1,orientation='horizontal')
    im2 = ax2.imshow(np.imag(image),cmap='seismic')
    ax2.set_title('Imaginary')
    plt.colorbar(im2,orientation='horizontal')
    return

def plot_eigen_probes(shared_probe, eigen_probe,method='angle'):
    methods = {
        'abs': [np.abs,'viridis'],
        'angle': [np.angle, 'seismic']
        }
    ims = []
    fig, ax = plt.subplots(1,eigen_probe.shape[1]+1, figsize=(5,5))
    im1= ax[0].imshow(methods[method][0](shared_probe[0,0,0]),cmap=methods[method][1])
    plt.colorbar(im1,orientation='horizontal')
    ax[0].set_title(f'Mode {0}')
    for i in range(eigen_probe.shape[1]):
        im = ax[1+i].imshow(methods[method][0](eigen_probe[0,i,0]),cmap=methods[method][1])
        ax[1+i].set_title(f'Mode {1+i}')
        ims.append(im)
        plt.colorbar(ims[i],orientation='horizontal')
    return 
        
def plot_shared_probes(shared_probe,method='angle'):
    methods = {
        'abs': [np.abs,'viridis'],
        'angle': [np.angle, 'seismic']
        }
    ims = []
    shared_probe = shared_probe.reshape((-1,)+shared_probe.shape[-3:])
    shared_probe = shared_probe[0]
    probeshape = shared_probe.shape
    shared_probe = np.concatenate(shared_probe, axis=0).transpose()
    plt.figure(figsize=(probeshape[0],3))
    img = plt.imshow(methods[method][0](shared_probe),cmap=methods[method][1])
    cbar = plt.colorbar(img,orientation='horizontal',)
    cbar.ax.invert_xaxis()
    # fig, ax = plt.subplots(1,shared_probe.shape[0],figsize=(14,3))
    # for i in range(shared_probe.shape[0]):
    #     im = ax[i].imshow(methods[method][0](shared_probe[i]),cmap=methods[method][1])
    #     ax[i].set_title(f'Mode {i}')
    #     ims.append(im)
    #     plt.colorbar(ims[i],orientation='horizontal')
    # plt.tight_layout()
    return

def plot_eigen_weights(eigen_weight,xaxis=None):
        plt.figure()
        for mode in range(eigen_weight.shape[1]):
            if type(xaxis) == np.ndarray:
                plt.plot(xaxis,eigen_weight[:,mode,0],label=f'Eigen mode: {mode}')
            else: 
                plt.plot(eigen_weight[:,mode,0],label=f'Eigen mode: {mode}')
            plt.xlabel('Position number')
            plt.ylabel('Weight')
            plt.legend()
            plt.tight_layout()
        return 
    
def plot_opr_weights(opr_mode_weights):
    plt.figure()
    for mode in range(opr_mode_weights.shape[1]):
        plt.plot(opr_mode_weights[:,mode],label=f'OPR mode: {mode+1}')
        plt.xlabel('Position number')
        plt.ylabel('OPR weight')
        plt.legend()
        plt.tight_layout()
    
def plot_intensity_map(scan,psi,probe):
    #TODO: this assumes the one shared probe at every position. include shared and eigen probes
    assert len(psi.shape) == 2, 'The psi should have a shape of length 2'
    assert len(probe.shape) == 2, 'The probe should have a shape of length 2'
    intensity_map = np.zeros_like(psi).real
    for row, col in scan:
        rowint, rowshift = round(row), row-round(row)
        colint, colshift = round(col), col-round(col)
        probeshift = shift(np.abs(probe)**2,(rowshift,colshift))
        intensity_map[rowint:rowint+probe.shape[-1],colint:colint+probe.shape[-2]] += probeshift
    
    plt.figure(figsize=(4,3))
    plt.imshow(intensity_map/np.max(intensity_map),cmap='nipy_spectral')
    plt.title('Intensity map')
    plt.colorbar()
    plt.axis(False) 
    return intensity_map

def plot_positions(true_pos, *args):
    """Taken from Tike python module at tike.view"""
    keys = ['true']
    plt.scatter(
        true_pos[:,0],
        true_pos[:,1],
        marker='+',
        color='black',
    )
    if len(args) > 1:
        plt.scatter(
            args[-1][:,0],
            args[-1][:,1],
            marker='o',
            color='red',
            facecolor='None',
        )
        keys.append('current')
    if len(args) >0:
        plt.scatter(
            args[0][:,0],
            args[0][:,1],
            marker='^',
            color='blue',
            facecolor='None',
        )
        keys.append('initial')
    plt.axis('equal')
    plt.legend(keys)
    
    if len(args) > 0 :
        lines = zip(true_pos,args[-1])
        lc = collections.LineCollection(lines, color='red',linestyle='dashed')
        plt.gca().add_collection(lc)
    
    for i in range(len(args)-1,0,-1):
        lines = zip(args[i], args[i-1])
        lc = collections.LineCollection(lines, color='red',linestyle='dashed')
        plt.gca().add_collection(lc)
        
    
    



def show_images(data):
    from ipywidgets import Button, HBox, VBox
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import numpy as np
    
    images = data  # Replace with your own numpy arrays
    current_image_index = 0  # Use a local variable instead of self.current_image_index
    
    def show_image(index):
        clear_output(wait=True)
        plt.figure(figsize=(4, 4))
        plt.imshow(np.log(images[index]))
        plt.title(f'Image {index}')  # Use index parameter to show the correct image index
        plt.tight_layout()
        plt.show()
        display(vbox)
    
    def on_next_button_clicked(b):
        nonlocal current_image_index
        current_image_index = (current_image_index + 1) % len(images)
        show_image(current_image_index)
    
    def on_prev_button_clicked(b):
        nonlocal current_image_index
        current_image_index = (current_image_index - 1) % len(images)
        show_image(current_image_index)
    
    # Create buttons
    next_button = Button(description="Next")
    prev_button = Button(description="Previous")
    
    # Assign button click events
    next_button.on_click(on_next_button_clicked)
    prev_button.on_click(on_prev_button_clicked)
    
    # Arrange buttons in a horizontal box
    hbox = HBox([prev_button, next_button])
    
    # Arrange everything in a vertical box
    vbox = VBox([hbox])
    
    # Display the first image initially
    show_image(current_image_index)
    
    # Optionally, you can return the VBox if needed
    return vbox

if __name__ == '__main__':
    None