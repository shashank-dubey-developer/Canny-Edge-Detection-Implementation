
#Importing all the important modules 
from cmath import pi           
from email.mime import image
from logging import exception
from PIL import Image
from numpy import asarray
import numpy as np
import scipy as sc
# import matplotlib.pylab as plt
import matplotlib.pyplot as plt
import sys




######################## FUNCTIONS #######################################################################################


def gaussian(kernal, sigma):
    ''' 1D gaussian kernel for smoothing  '''
    x = kernal//2
    xRange = np.arange(-x, (x+1), 1)
    gauss = np.exp((-(xRange**2))/(2*(sigma**2))) / (np.sqrt(2*np.pi)*sigma)
    return gauss

def gaussianDerivative(kernal, sigma):
    ''' 1D gaussian derivative to get teh respective components, with respect to axis  '''
    x = kernal//2
    xRange = np.arange(-x, (x+1), 1)
    gaussD = -((xRange * np.exp(-(xRange**2)/(2*(sigma**2))))/(np.sqrt(2*np.pi)*sigma**3))
    return gaussD

def convolution(image, kernelFilter): 
    ''' this function convolves image and kernel  '''
    size = int(np.max(kernelFilter.shape)/2)
    output = np.zeros_like(image)
    image_padded = np.zeros((image.shape[0] + 2*size, image.shape[1] + 2*size))
    padsize =2*(size)-1
    image_padded[1:-padsize, 1:-padsize] = image
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x]=(kernelFilter * image_padded[y: (y+kernelFilter.shape[0]), x:(x+kernelFilter.shape[1]) ]).sum()
    return output

def minmaxNormalisation(image):
    ''' ranging the pixel value between 0 and 255  '''
    lowVal = image.min()
    highVal = image.max()
    image = (image-lowVal)*1.0/(highVal-lowVal)
    return image*255

def magnitude(I_X, I_Y):
    ''' calculates the magnitude of the images with respect to axis  '''
    result = np.hypot(I_X,I_Y)
    return result        

def non_Max_Supression_Algo(image, I_X,I_Y):
    ''' this function thins the edges  '''
    theta = np.arctan2(I_Y,I_X)
    angle = theta * 180 / np.pi
    angle[angle<0] += 180
    
    P,Q = image.shape
    output = np.zeros((P,Q), dtype=np.int32)

    for i in range(1, P-1):
        for j in range (1, Q-1):
            l = 255
            m = 255

            #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                l = image[i, j+1]
                m = image[i, j-1]
               #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                l = image[i+1, j-1]
                m = image[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                l = image[i+1, j]
                m = image[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                l = image[i-1, j-1]
                m = image[i+1, j+1]
            if (image[i,j] >= m) and (image[i,j] >= l):
                output[i,j] = image[i,j]
            else:
                output[i,j] = 0
    return output

def threshold_function(I, lowThresRatio=0.05, highThresRatio=0.2):
    ''' this function eliminates weak edges '''
    
    highThres = I.max() * highThresRatio
    lowThres = highThres * lowThresRatio
    
    row, col = I.shape
    result = np.zeros((row,col), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(I >= highThres)
    zeros_i, zeros_j = np.where(I < lowThres)
    
    weak_i, weak_j = np.where((I <= highThres) & (I >= lowThres))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    return (result, weak, strong)

def hysteresis(I, weakThreshold, strongThreshold=255):
    ''' this function eliminates connect connected strong edges  '''
    row, col = I.shape  
    for i in range(1, row-1):
        for j in range(1, col-1):
            if I[i,j] == weakThreshold:
                try:
                    if ((I[i+1, j-1] == strongThreshold) or (I[i+1, j] == strongThreshold) 
                        or (I[i+1, j+1] == strongThreshold) or (I[i, j-1] == strongThreshold) or (I[i, j+1] == strongThreshold)
                        or (I[i-1, j-1] == strongThreshold) or (I[i-1, j] == strongThreshold) or (I[i-1, j+1] == strongThreshold)):
                        
                        I[i, j] = strongThreshold

                    else:

                        I[i, j] = 0


                except IndexError as ex:
                    pass


    return I

######################### FUNCTIONS #####################################################################################

#Reading the gray Scale Image from "Berkley Segmentation Datasets, Training Images", stored in Matrix I
#Kernal stored as integer and Sigma Value stored in list
I, kernal, sigma = asarray(Image.open(r"sample image.jpg")), 3, [1,1.5,2]

#traversing for each sigma value in the list "sigma"
for sigmaval in sigma:

    # Creating 1-D Gaussian mask and Gaussian Derivative mask.
    gaussValue, gaussDerivative = gaussian(kernal, sigmaval).reshape(1,-1), gaussianDerivative(kernal, sigmaval).reshape(1,-1) 

    # Calculating (Ix, Iy) using gaussian kernal.
    Ix, Iy = convolution(I, gaussValue), (convolution(I, gaussValue.T))

    # Normalising both (Ix,Iy).
    Ix, Iy = minmaxNormalisation(Ix), minmaxNormalisation(Iy)

    # Calculating I_X(Ix') & I_Y(Iy') using gaussian derivative kernal.
    I_X, I_Y = convolution(Ix, gaussDerivative), (convolution(Iy, gaussDerivative.T))

    # Calculating Image magnitude
    # Hypot function is sqrt(Ix**2+Iy**2)
    Mxy = magnitude(I_X, I_Y)

    # Applying non max supression to the image magnitude
    nonMaxSupressedImage = non_Max_Supression_Algo(Mxy, I_X, I_Y)

    # Calculating weak and strong thresholds and passing it to the hysteresis
    threshold, weak, strong = threshold_function(nonMaxSupressedImage)
    outputImage = hysteresis(threshold, weak, strong)
    # Showing then final oputput image
    plt.imshow(outputImage, cmap = 'gray')
    plt.show()
    









