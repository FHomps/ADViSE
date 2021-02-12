import numpy as np
from scipy import ndimage
import torch

def meanFilter(array, kernelSize):
    return ndimage.convolve(array, np.full((kernelSize, kernelSize), 1.0 / (kernelSize ** 2)))

# Transforms an image of the slope into a map of where the rover can reasonably go.
# The slope threshold is in degrees.
def getViabilityMap(img, slopeThreshold=25, KSize_mean=5, KSize_median=5):
    img_np = img.detach().numpy()
    img_np = meanFilter(img_np, KSize_mean)
    img_np = np.where(img_np > slopeThreshold / 45, True, False)
    img_np = ndimage.median_filter(img_np, size=KSize_median)
    print(type(img_np))
    return img_np