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

def saveSample(model, dataset, filename, n_samples, n_cols = 6, normalize = False, batch_index = True):
    selection = random.sample(range(len(dataset)), n_samples)
    if batch_index:
        inp, gt = dataset[selection]
        out = model(inp).cpu()
        samples = torch.cat((inp, out, gt), -2)
    else:
        samples = []
        for idx in selection:
            inp, gt = dataset[idx]
            out = model(inp[None])[0].cpu()
            samples.append(torch.cat((inp, out, gt), -2))

    torchvision.utils.save_image(samples, filename, nrow=n_cols, normalize=normalize)
    