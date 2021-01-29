import torchvision.utils
import torch
import random

class StatTracker():
    def __init__(self, smoothing=True, smoothing_factor = 0.5):
        self.items = list()
        self.smoothing = smoothing
        if smoothing:
            self.alpha = smoothing_factor
            self.smooth = list()

    def __len__(self):
        return len(self.items)

    def log(self, x):
        self.items.append(x)
        if self.smoothing:
            if len(self.items) == 1:
                self.smooth.append(x)
            else:
                self.smooth.append(self.alpha * x + (1 - self.alpha) * self.smooth[-1])

def saveSample(model, dataset, filename, n_samples, n_cols = 6, normalize = False):
    inp, gt = dataset[random.sample(range(len(dataset)), n_samples)]
    out = model(inp).cpu()
    samples = torch.cat((inp, out, gt), -2)
    torchvision.utils.save_image(samples, filename, nrow=n_cols, normalize=normalize, )
    