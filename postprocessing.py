import numpy as np
from scipy import ndimage
import torch


# Transforms an image of the slope into a map of where the rover can reasonably go.
# The slope threshold is in degrees.
def getViabilityMap(img, slopeThreshold=25, K_mean=5, percent_spread=50, K_spread=5):
    if torch.is_tensor(img):
        img_np = img.cpu().detach().numpy()
    else:
        img_np = img
    img_np = ndimage.uniform_filter(img_np, size=(1, 1, K_mean, K_mean))
    img_np = np.where(img_np > slopeThreshold / 45, True, False)
    img_np = np.logical_or(ndimage.percentile_filter(img_np, percent_spread, size=(1, 1, K_spread, K_spread)), img_np)
    return img_np

class ViabilityLoss:
    def __init__(self, slopeThreshold=25, K_mean=5, percent_spread=50, K_spread=5):
        self.slopeThreshold = slopeThreshold
        self.K_mean = K_mean
        self.percent_spread = percent_spread
        self.K_spread = K_spread
    
    def __call__(self, out, gt):
        return np.mean(np.bitwise_xor(
            getViabilityMap(out.detach().cpu().numpy(), self.slopeThreshold, self.K_mean, self.percent_spread, self.K_spread),
            getViabilityMap(gt.detach().cpu().numpy(), self.slopeThreshold, self.K_mean, self.percent_spread, self.K_spread)
        ))
       