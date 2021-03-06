import numpy as np
import torch
from torch.nn.functional import conv2d, unfold, pad, mse_loss
import torchvision
from torchvision.transforms.functional import resize
import random

def gaussian_blur(img, ksize=3, strength=1):
    weights = img.new_empty((1, 1, ksize, ksize), dtype=torch.float, requires_grad=False)
    
    center = (ksize-1) / 2
    
    coeffs = {}
    for i in range(ksize):
        for j in range(ksize):
            dist = (i - center)**2 + (j - center)**2
            if dist not in coeffs.keys():
                coeffs[dist] = np.exp(-dist / (2 * strength))
            weights[0][0][i][j] = coeffs[dist]
    
    weights *= 1 / torch.sum(weights)
    if ksize % 2 != 0:
        pad_size = (int(ksize / 2),)*4
    else:
        pad_size = (int(ksize / 2) - 1, int(ksize / 2)) * 2
    return conv2d(pad(img, pad_size, mode='replicate'), weights)

def quantile_filter(img, ksize=3, quantile=.5):
    if ksize % 2 != 0:
        pad_size = (int(ksize / 2),)*4
    else:
        pad_size = (int(ksize / 2) - 1, int(ksize / 2)) * 2
    img_unf = unfold(pad(img, pad_size, value=np.nan), ksize)
    return img_unf.nanquantile(quantile, 1, keepdim=True).reshape(img.size())

def quantile_filter_inplace(img, ksize=3, quantile=.5):
    if ksize % 2 != 0:
        pad_size = (int(ksize / 2),)*4
    else:
        pad_size = (int(ksize / 2) - 1, int(ksize / 2)) * 2
    img_unf = unfold(pad(img, pad_size, value=np.nan), ksize)
    flat_img = img.view(*img.size()[:2], img.size(2) * img.size(3))
    torch.nanquantile(img_unf, quantile, 1, keepdim=True, out=flat_img)


# Transforms a tensor of the slope into a map of where the rover can reasonably go.
# tresholds should be a list of critical slopes in degrees (0 to 45)

def getViabilityMap(slope_tensor):
    thresholds=[15, 25]
    is_byte_img = isinstance(slope_tensor, torch.ByteTensor)

    if is_byte_img:
        out = slope_tensor.float() / 255
    else:
        out = torch.clone(slope_tensor)
        
    in_res = slope_tensor.size(-1)
        
    out = resize(out, round(in_res / 2))
    quantile_filter_inplace(out, 6, .75)
    out = resize(out, in_res)
    out = gaussian_blur(out, 5)
    
    if thresholds == None:
        return out if not is_byte_img else (out * 255).byte()
    
    threshold_value_increment = 1 / len(thresholds)
    vmap = torch.zeros_like(out)
    
    for t in sorted(thresholds):
        vmap += (out > t / 45) * threshold_value_increment
    
    if is_byte_img:
        vmap = (vmap * 255).byte()
    
    return vmap

def saveProcessedSample(tensor, filename, n_samples = 36, n_cols = 12, normalize = False, seed=None):
    if seed == None:
        random.seed(None)
        torch.manual_seed(random.getrandbits(64))
    else:
        random.seed(seed)
        torch.manual_seed(seed)
    
    if (tensor.size(0) > n_samples):
        selection = random.sample(range(tensor.size(0)), n_samples)
    else:
        selection = [i for i in range(tensor.size(0))]
    
    img_in = tensor[selection]
    print("Computing VMap sample...")
    img_out = getViabilityMap(img_in)
    samples = torch.cat((img_in, img_out), -2)
    
    print("Saving VMap sample...")
    torchvision.utils.save_image(samples.float() / 255, filename, nrow=n_cols, normalize=normalize)

class ViabilityLoss:
    def __call__(self, out, gt):
        return mse_loss(getViabilityMap(out.detach()), gt)
