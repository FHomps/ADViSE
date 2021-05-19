import random
import numpy as np
import collections

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import torch
from torch import tensor

class LandingZoneDataset(Dataset):
    def __init__(self, img_tensor, gt_tensor, selection, outRes = 128,
                 minSizeRatio = 1/3, maxSizeRatio = 1., rotate = True, compensateCorners = True, twin_img_tensor = None):
        
        self.selection = tensor(selection, dtype=torch.int64)
        self.images = torch.index_select(img_tensor, 0, self.selection)
        if twin_img_tensor != None:
            self.images = torch.cat((self.images, torch.index_select(twin_img_tensor, 0, self.selection)), 0)
        self.groundTruth = torch.index_select(gt_tensor, 0, self.selection)
        if twin_img_tensor != None:
            self.groundTruth = self.groundTruth.repeat((2, 1, 1, 1))
        
        self.img_size = self.images.size()[-1]
        self.gt_size = self.groundTruth.size()[-1]

        self.outRes = outRes
        self.RRC_scale = (minSizeRatio**2, maxSizeRatio**2)
        if rotate:
            if compensateCorners:
                self.transform = self.transform_compensated_rotate
            else:
                self.transform = self.transform_rotate
        else:
            self.transform = self.transform_noRotate
        
    def transform_noRotate(self, image, image_gt):
        RCParams_gt = np.array(transforms.RandomResizedCrop.get_params(image_gt, self.RRC_scale, (1., 1.)))
        RCParams_img = RCParams_gt * round(self.img_size / self.gt_size)
        
        return (TF.resize(TF.crop(image, *RCParams_img), self.outRes).float() / 255,
                TF.resize(TF.crop(image_gt, *RCParams_gt), self.outRes).float() / 255)
        
    def transform_rotate(self, image, image_gt):
        RCParams_gt = np.array(transforms.RandomResizedCrop.get_params(image_gt, self.RRC_scale, (1., 1.)))
        RCParams_img = RCParams_gt * round(self.img_size / self.gt_size)
        angle = random.random() * 180
        
        return (TF.rotate(TF.resize(TF.crop(image, *RCParams_img), self.outRes), angle, resample=Image.BILINEAR).float() / 255,
                TF.rotate(TF.resize(TF.crop(image_gt, *RCParams_gt), self.outRes), angle, resample=Image.BILINEAR).float() / 255)

    def transform_compensated_rotate(self, image, image_gt):
        if image.dim() == 3:
            channels_img, _, img_res = image.size()
            channels_gt, _, gt_res = image_gt.size()
        else:
            batchSize, channels_img, _, img_res = image.size()
            _, channels_gt, _, gt_res = image_gt.size()

        sizeRatio = round(img_res / gt_res)
        
        # RandomResizedCrop returns a crop within the bounds of the image
        i, j, h, w = transforms.RandomResizedCrop.get_params(image_gt, self.RRC_scale, (1., 1.))

        # The crop boundaries are expanded by an optimal margin to include more data and minimize black corners after rotation
        angle = random.random() * 180
        alpha = (45 - abs((angle % 90) - 45)) * np.pi / 180
        expansionRatio = np.cos(alpha) + np.sin(alpha)
        cropMargin = int((expansionRatio - 1) * w / 2 + 1)

        i -= cropMargin
        j -= cropMargin
        h += 2 * cropMargin
        w += 2 * cropMargin

        # pytorch / torchvision's crop doesn't support out-of-bounds coordinates, so we have to do it manually.

        # Coordinates from which the pixels are copied in the original gt
        copyBox_gt = np.array([max(i, 0), min(gt_res, i + h), max(j, 0), min(gt_res, j + w)])
        # Coordinates in which the pixels are pasted in the cropped gt
        pasteBox_gt = np.array([max(-i, 0), min(gt_res - i, h), max(-j, 0), min(gt_res - j, w)])
        # Same for the image
        copyBox_img = copyBox_gt * sizeRatio
        pasteBox_img = pasteBox_gt * sizeRatio

        if (image.dim() == 3):
            cropped_gt = image_gt.new_zeros((channels_gt, h, w))
            cropped_img = image.new_zeros((channels_img, h * sizeRatio, w * sizeRatio))

            cropped_gt[:, pasteBox_gt[0]:pasteBox_gt[1], pasteBox_gt[2]:pasteBox_gt[3]] = \
                image_gt[:, copyBox_gt[0]:copyBox_gt[1], copyBox_gt[2]:copyBox_gt[3]]
            cropped_img[:, pasteBox_img[0]:pasteBox_img[1], pasteBox_img[2]:pasteBox_img[3]] = \
                image[:, copyBox_img[0]:copyBox_img[1], copyBox_img[2]:copyBox_img[3]]
        else:
            cropped_gt = image_gt.new_zeros((batchSize, channels_gt, h, w))
            cropped_img = image.new_zeros((batchSize, channels_img, h * sizeRatio, w * sizeRatio))

            cropped_gt[:, :, pasteBox_gt[0]:pasteBox_gt[1], pasteBox_gt[2]:pasteBox_gt[3]] = \
                image_gt[:, :, copyBox_gt[0]:copyBox_gt[1], copyBox_gt[2]:copyBox_gt[3]]
            cropped_img[:, :, pasteBox_img[0]:pasteBox_img[1], pasteBox_img[2]:pasteBox_img[3]] = \
                image[:, :, copyBox_img[0]:copyBox_img[1], copyBox_img[2]:copyBox_img[3]]
    
        resMargin = int(((expansionRatio - 1) * self.outRes) / 2 + 1)
        expandedRes = self.outRes + resMargin * 2
        FinalCropParams = (resMargin, resMargin, self.outRes, self.outRes)
        
        return (TF.crop(TF.rotate(TF.resize(cropped_img, expandedRes), angle, resample=Image.BILINEAR), *FinalCropParams).float() / 255,
                TF.crop(TF.rotate(TF.resize(cropped_gt, expandedRes), angle, resample=Image.BILINEAR), *FinalCropParams).float() / 255)


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