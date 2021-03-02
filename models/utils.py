import torchvision.utils
import torch
import random

class StatTracker():
    def __init__(self, smoothing_factor=None):
        self.items = list()
        self.do_smoothing = (smoothing_factor != None)
        if self.do_smoothing:
            self.alpha = min(1, max(0, 1-smoothing_factor))
            self.smooth = list()

    def __len__(self):
        return len(self.items)

    def get_dict(self):
        d = {
            "items": self.items.copy(),
            "do_smoothing": self.do_smoothing
        }
        if self.do_smoothing:
            d["alpha"] = self.alpha
            d["smooth"] = self.smooth.copy()
        return d

    @staticmethod
    def from_dict(d):
        tracker = StatTracker()
        tracker.items = d["items"].copy()
        tracker.do_smoothing = d["do_smoothing"]
        if tracker.do_smoothing:
            tracker.alpha = d["alpha"]
            tracker.smooth = d["smooth"].copy()
        return tracker

    def log(self, x):
        if (torch.is_tensor(x)):
            x = x.item()
        self.items.append(x)
        if self.do_smoothing:
            if len(self.items) == 1:
                self.smooth.append(x)
            else:
                self.smooth.append(self.alpha * x + (1 - self.alpha) * self.smooth[-1])

def saveSample(model, dataset, filename, n_samples, n_cols = 6, normalize = False, batch_index = True, seed=None):
    random.seed(seed)
    torch.manual_seed(seed)
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
    random.seed(None)
    torch.manual_seed(random.getrandbits(64))
    
    