# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 09:10:28 2023

@author: taylo
"""

import numpy as np
from sklearn.neighbors import BallTree

def scan(pattern, center,width,height,step):
    scan = STYLE_DICT[pattern]
    return scan(center, width, height, step)


def raster(center, width, height, step):
    # Number of points that fit in each direction
    nx = int(np.floor(width / step)) + 1
    ny = int(np.floor(height / step)) + 1

    # Symmetric about center, exact spacing = step
    x = center[0] + (np.arange(nx) - (nx - 1) / 2) * step
    y = center[1] + (np.arange(ny) - (ny - 1) / 2) * step

    X, Y = np.meshgrid(x, y)
    for i in range(X.shape[0]):
        # Flip every other row to make a snake-like (serpentine) pattern
        if i % 2:
            X[i] = np.flip(X[i])

    return np.column_stack((X.ravel(), Y.ravel()))

def concentric_scan(center,width,height,step):
    """
    Concentric-circle scan positions filling a width x height rectangle.

    center : (x0, y0)
    width  : rectangle width
    height : rectangle height
    step   : spacing between rings and (approximately) between points on a ring
    Returns an (N, 2) array of (x, y) positions.
    """
    r_max = 0.5 * np.hypot(width, height)      # circle covering the rectangle's corners
    n_rings = int(np.ceil(r_max / step))

    pts = [(0.0, 0.0)]
    for k in range(1, n_rings + 1):
        r = k * step
        n = 6 * k                              # 6k points on ring k -> arc spacing ~ step
        theta = 2 * np.pi * np.arange(n) / n
        pts.extend(zip(r * np.cos(theta), r * np.sin(theta)))

    pts = np.array(pts)
    keep = (np.abs(pts[:, 0]) <= width / 2) & (np.abs(pts[:, 1]) <= height / 2)
    return pts[keep] + np.asarray(center)                                
                            
def spiral(center, width, height, step):
    positions= []
    th = 0
    r = step/4
    positions = [[r,th]]
    rmax = np.sqrt(width**2+height**2)/2
    while r<rmax:
        th = th + step/r
        r = step/(2*np.pi)*th
        x = r*np.cos(th)
        y = r*np.sin(th)
        positions.append([x,y])
    positions = np.array(positions)+center
    positions = positions[
        (abs(positions[:, 0]) <= width / 2) & (abs(positions[:, 1]) <= height / 2)
    ]    
    return positions

def fermat_spiral(center, width, height, step):
    r_max = np.sqrt(width**2 + height**2) / 2
    c = step/1.6917708026774263 #step is linearly propto the c scale
    n = 0
    r = 0
    positions = []
    while r<r_max:
        r = c*np.sqrt(n)
        theta  = n*137.508*np.pi/180
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        positions.append([x,y])
        n+=1
    
    positions = np.array(positions)+center
    
    positions = positions[
        (abs(positions[:, 0]) <= width / 2) & (abs(positions[:, 1]) <= height / 2)
    ]
    
    positions = minimize_total_distance(positions)
    return positions

def minimize_total_distance(positions):
    tree = BallTree(positions, metric='euclidean')
    n = len(positions)
    visited = np.zeros(n,dtype=bool)
    ordered_points = [0]
    visited[0] = True 
    for _ in range(n-1):
        last = ordered_points[-1]
        _, nearest = tree.query((positions[last]).reshape(1,-1),k=n)
        for idx in nearest[0]:
            if not visited[idx]:
                ordered_points.append(idx)
                visited[idx] = True
                break
    return positions[ordered_points]
    

STYLE_DICT = {'raster': raster, 'spiral': spiral, 'concentric': concentric_scan, 'fermat': fermat_spiral}

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    positions= scan('concentric',(0.0,0.0),15e-6,15e-6,1e-6)
    positions[:,0] -= np.median(positions[:,0])
    positions[:,1] -= np.median(positions[:,1])
    maxX, maxY = positions[:,1].max(), positions[:,0].max() 
    positions = positions[
        (positions[:, 0]-maxY/2 <= 33e-6/2) & (positions[:, 1]-maxY/2 <= 33e-6/2)
    ]
    plt.plot(positions[:,0]*10**6,positions[:,1]*10**6,'-o')
    plt.gca().set_aspect('equal')
    print(positions.shape)
    
    
    
    
    
        