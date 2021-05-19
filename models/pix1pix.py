import torch.nn as nn
import torch
import numpy as np
import pickle

from .model import Model
from .utils import StatTracker

def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


class Pix1Pix(Model):
    def __init__(self, device, img_res, in_channels=1, out_channels=1, learning_rate=0.0002, b1=0.5, b2=0.999, extra_losses={}, use_MSE=False):
        super(Pix1Pix, self).__init__()
        
        if use_MSE:
            self.criterion_pixelwise = torch.nn.MSELoss().to(device)
        else:
            self.criterion_pixelwise = torch.nn.L1Loss().to(device)
        
        self.generator = GeneratorUNet(in_channels, out_channels).to(device)
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(b1, b2))
        self.Tensor = torch.cuda.FloatTensor if device.type=="cuda" else torch.FloatTensor
        
        self.extra_losses = extra_losses
        self.initWeights()
    
    def train(self, inp, label, compute_extra_losses=True):
        self.generator.train()
        
        real_in = inp.type(self.Tensor)
        real_out = label.type(self.Tensor)
        
        self.optimizer_G.zero_grad()
        
        fake_out = self.generator(real_in)
        loss_G = self.criterion_pixelwise(fake_out, real_out)
        loss_G.backward()
        
        self.optimizer_G.step()
        losses = {}
        
        losses["GLoss_pixel"] = loss_G.item()
        
        if compute_extra_losses:
            for k, l in self.extra_losses.items():
                losses[k] = l(fake_out, real_out).item()
        
        return losses
                
    def evaluate(self, inp, label, compute_extra_losses=True):
        self.generator.eval()
        
        with torch.no_grad():
            real_in = inp.type(self.Tensor)
            real_out = label.type(self.Tensor)

            fake_out = self.generator(real_in)
            loss_G = self.criterion_pixelwise(fake_out, real_out)
            
            losses = {}
            
            losses["GLoss_pixel"] = loss_G.item()
            
            if compute_extra_losses:
                for k, l in self.extra_losses.items():
                    losses[k] = l(fake_out, real_out).item()
            
            return fake_out, losses
    
    def saveToFile(self, filename_stub):
        torch.save(self.generator.state_dict(), filename_stub + ".ts")
    
    def loadFromFile(self, filename_stub):
        self.generator.load_state_dict(torch.load(filename_stub + ".ts"))
        
    def initWeights(self):
        self.generator.apply(weights_init_normal)
        
    def __call__(self, inp):
        self.generator.eval()
        with torch.no_grad():
            return self.generator(inp.type(self.Tensor))