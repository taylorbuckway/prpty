# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:54:52 2024

@author: tbuckway
"""


import numpy as np

def random_pad(images,pad_width,low,high):
    mask = np.pad(np.zeros_like(images),pad_width,constant_values=1)
    images = np.pad(images,pad_width,constant_values=0)
    random_array = np.random.randint(low=low,high=high,size=images.shape[-2:])*mask
    return images+random_array
    
    
 
if __name__ in '__main__':
    import time
    
    array = np.zeros((2,2048,2048))
    pad_width = ((0,0),(10,10),(10,10))
    x = time.time()
    padded_array = random_pad(array,pad_width,0,10)
    print(padded_array)