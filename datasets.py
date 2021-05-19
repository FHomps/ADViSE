import random
import numpy as np
import collections

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import torch
from torch import tensor

class LandingZoneDataset(Dataset):
    def __init__(self, img_tensor, gt_tensor, selection, outRes = 128,
                 minZoom = 0.125, maxZoom = 8, rotate = True, twin_img_tensor = None):
        
        self.selection = tensor(selection, dtype=torch.int64)
        self.images = torch.index_select(img_tensor, 0, self.selection)
        self.groundTruth = torch.index_select(gt_tensor, 0, self.selection)
        if twin_img_tensor != None:
            self.images = torch.cat((self.images, torch.index_select(twin_img_tensor, 0, self.selection)), 0)
            self.groundTruth = self.groundTruth.repeat((2, 1, 1, 1))
        
        self.img_size = self.images.size()[-1]
        self.gt_size = self.groundTruth.size()[-1]

        self.outRes = outRes
        self.minZoom = minZoom
        self.maxZoom = maxZoom
        if rotate:
            self.transform = self.transform_rotate
        else:
            self.transform = self.transform_norotate
    
    def transform_rotate(self, image, image_gt):
        if image.dim() == 3:
            channels_img, _, img_res = image.size()
            channels_gt, _, gt_res = image_gt.size()
        else:
            batchSize, channels_img, _, img_res = image.size()
            _, channels_gt, _, gt_res = image_gt.size()

        img_gt_ratio = round(img_res / gt_res)
        
        z = np.exp(random.uniform(np.log(self.minZoom), np.log(self.maxZoom)))

        # Compute extended crop boundaries (with margin for rotate)
        
        # First, get standard crop boundaries.
        if z < 1:
            b_h = b_w = round(gt_res / z)
            b_t = b_l = round((gt_res - b_h) / 2)
        else:
            # RandomResizedCrop returns a crop within the bounds of the image
            RRCScale = 1 / (z * z)
            b_t, b_l, b_h, b_w = T.RandomResizedCrop.get_params(image_gt, (RRCScale, RRCScale), (1., 1.))
            
        # The crop boundaries are extended by an optimal margin to include more data and minimize black corners after rotation
        angle = random.random() * 360
        alpha = (45 - abs((angle % 90) - 45)) * np.pi / 180
        expansionRatio = np.cos(alpha) + np.sin(alpha)
        cropMargin = int((expansionRatio - 1) * b_w / 2 + 1)

        eb_t = b_t - cropMargin
        eb_l = b_l - cropMargin
        eb_b = eb_t + b_h + 2 * cropMargin
        eb_r = eb_l + b_w + 2 * cropMargin
        
        # Get restricted bounds: intersection of extended bounds and image bounds
        rb_t = max(eb_t, 0)
        rb_l = max(eb_l, 0)
        rb_b = min(eb_b, gt_res)
        rb_r = min(eb_r, gt_res)
        
        gt = TF.crop(image_gt, rb_t, rb_l, rb_b - rb_t, rb_r - rb_l)
        img = TF.crop(image, *map(lambda x : round(x * img_gt_ratio), (rb_t, rb_l, rb_b - rb_t, rb_r - rb_l)))
        
        # Get the scaled down bounds
        cropMargin, eb_t, eb_l, eb_b, eb_r, rb_t, rb_l, rb_b, rb_r = map(lambda x : round(x * self.outRes / b_w),
       (cropMargin, eb_t, eb_l, eb_b, eb_r, rb_t, rb_l, rb_b, rb_r))
        
        # Scale down the crop
        img = TF.resize(img, (rb_b - rb_t, rb_r - rb_l), interpolation=Image.BILINEAR)
        gt = TF.resize(gt, (rb_b - rb_t, rb_r - rb_l), interpolation=Image.BILINEAR)
        
        # Reflect pad up to extended boundaries
        if image.dim() == 3:
            pad_size = ((0, 0), (rb_t - eb_t, eb_b - rb_b), (rb_l - eb_l, eb_r - rb_r))
        else:
            pad_size = ((0, 0), (0, 0), (rb_t - eb_t, eb_b - rb_b), (rb_l - eb_l, eb_r - rb_r))
        # PyTorch's pad doesn't support reflect padding on big areas, so numpy is used
        img = torch.from_numpy(np.pad(img.numpy(), pad_size, mode='wrap'))
        gt = torch.from_numpy(np.pad(gt.numpy(), pad_size, mode='wrap'))
        
        # Now, do the rotation and crop to the final size
        img = TF.rotate(img, angle, resample=Image.BILINEAR)
        gt = TF.rotate(gt, angle, resample=Image.BILINEAR)
        
        img = TF.crop(img, cropMargin, cropMargin, self.outRes, self.outRes)
        gt = TF.crop(gt, cropMargin, cropMargin, self.outRes, self.outRes)
        
        return (img.float() / 255, gt.float() / 255)

    def transform_norotate(self, image, image_gt):
        if image.dim() == 3:
            channels_img, _, img_res = image.size()
            channels_gt, _, gt_res = image_gt.size()
        else:
            batchSize, channels_img, _, img_res = image.size()
            _, channels_gt, _, gt_res = image_gt.size()

        img_gt_ratio = round(img_res / gt_res)
        
        z = np.exp(random.uniform(np.log(self.minZoom), np.log(self.maxZoom)))
        
        if z >= 1:
            # RandomResizedCrop returns a crop within the bounds of the image
            RRCScale = 1 / (z * z)
            b_t, b_l, b_h, b_w = T.RandomResizedCrop.get_params(image_gt, (RRCScale, RRCScale), (1., 1.))
            img = TF.crop(image, *map(lambda x : round(x * img_gt_ratio), (b_t, b_l, b_h, b_w)))
            gt = TF.crop(image_gt, b_t, b_l, b_h, b_w)
            img = TF.resize(img, self.outRes)
            gt = TF.resize(gt, self.outRes)
            return (img.float() / 255, gt.float() / 255)
        
        else:
            # Get scaled down crop boundaries
            b_h = b_w = round(gt_res / z)
            b_t = b_l = round((gt_res - b_h) / 2)
            
            zone_res, b_h, b_w, b_t, b_l = map(lambda x : round(x * self.outRes / b_w),
           (gt_res,   b_h, b_w, b_t, b_l))
            
            # Scale down the image
            img = TF.resize(image, (zone_res, zone_res), interpolation=Image.BILINEAR)
            gt = TF.resize(image_gt, (zone_res, zone_res), interpolation=Image.BILINEAR)
            
            # Reflect pad up to extended boundaries
            lpad = round((self.outRes - zone_res) / 2)
            rpad = self.outRes - zone_res - lpad
            if image.dim() == 3:
                pad_size = ((0, 0), (lpad, rpad), (lpad, rpad))
            else:
                pad_size = ((0, 0), (0, 0), (lpad, rpad), (lpad, rpad))
            # PyTorch's pad doesn't support reflect padding on big areas, so numpy is used
            img = torch.from_numpy(np.pad(img.numpy(), pad_size, mode='wrap'))
            gt = torch.from_numpy(np.pad(gt.numpy(), pad_size, mode='wrap'))
            
            return (img.float() / 255, gt.float() / 255)

    def __len__(self):
        return self.images.size(0)
    
    def __getitem__(self, idx):
        return self.transform(self.images[idx], self.groundTruth[idx])

    def getAll(self, shuffle=True):
        if (shuffle):
            indices = torch.randperm(len(self.selection))
            return self.transform(self.images[indices], self.groundTruth[indices])
        else:
            return self.transform(self.images, self.groundTruth)