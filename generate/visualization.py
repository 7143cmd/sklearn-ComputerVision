import numpy as np
from skimage import io


# target #

target: int = 0

def read():
    with open('log.txt', 'r') as file:                
        for index, File in enumerate(file):           
            if index == target:                           
                File = ''.join(char if char.isdigit() else ' ' for char in File)     
                File = File.strip().split()                        
                File = np.array(File, dtype=np.uint8)                   
                reshaped = File.reshape((32,32,3))                     
                io.imsave('rework.png', reshaped)                          

read()