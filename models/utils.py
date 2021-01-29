import torchvision.utils
import torch
import random

class StatTracker():
    def __init__(self, smoothing_factor = None):
        self.items = list()
        if smoothing_factor != None:
            self.alpha = smoothing_factor
            self.smooth = list()
        else:
            self.smooth = None

    def __len__(self):
        return len(self.items)

    def log(self, x):
        self.items.append(x)
        if self.smooth != None:
            if len(self.items) == 1:
                self.smooth.append(x)
            else:
                self.smooth.append(self.alpha * x + (1 - self.alpha) * self.smooth[-1])

def saveSample(model, dataset, filename, n_samples, n_cols = 6, normalize = False):
    inp, gt = dataset[random.sample(range(len(dataset)), n_samples)]
    out = model(inp).cpu()
    samples = torch.cat((inp, out, gt), -2)
    torchvision.utils.save_image(samples, filename, nrow=n_cols, normalize=normalize, )
    