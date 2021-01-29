from torchvision.utils import save_image
import torch

class StatTracker():
    def __init__(self, sliding_avg_window=None):
        self.sums = list()
        self.avgs = list()
        self.items = list()
        self.window = sliding_avg_window
        if self.window != None:
            self.sliding_avgs = list()

    def __len__(self):
        return len(self.items)

    def log(self, x):
        self.items.append(x)
        self.sums.append(sum(self.items))
        self.avgs.append(self.sums[-1] / len(self.items))
        if self.window != None:
            self.sliding_avgs.append(sum(self.items[-self.window:]) / min(self.window, len(self.items)))

def saveSamples(model, dataset, filename, n_samples, n_cols = 8, normalize = False):
    inp, gt = dataset[n_samples]
    out = model(inp)
    samples = torch.cat((inp, out, gt), -2)
    save_image(samples, filename, nrow=n_cols, normalize=normalize)
    