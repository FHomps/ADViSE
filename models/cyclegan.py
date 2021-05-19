import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pickle
import itertools
import random

from .model import Model
from .utils import StatTracker

def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    elif class_name.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           RESNET
##############################


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, input_shape, num_residual_blocks):
        super(GeneratorResNet, self).__init__()

        channels = input_shape[0]

        # Initial convolution block
        out_features = 64
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_features, 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # Downsampling
        for _ in range(2):
            out_features *= 2
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Residual blocks
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(out_features)]

        # Upsampling
        for _ in range(2):
            out_features //= 2
            model += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, 3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(out_features, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        y = self.model(x)
        return y


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)
   
class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class CycleGAN(Model):
    def __init__(self, device, img_res, channels=1, learning_rate=0.0002, b1=0.5, b2=0.999, n_residual_blocks=9, lambda_id=5.0, lambda_cyc=10.0, extra_losses={}):
        super(CycleGAN, self).__init__()
        
        input_shape = (channels, img_res, img_res)
        self.G_AB = GeneratorResNet(input_shape, n_residual_blocks).to(device)
        self.G_BA = GeneratorResNet(input_shape, n_residual_blocks).to(device)
        self.D_A = Discriminator(input_shape).to(device)
        self.D_B = Discriminator(input_shape).to(device)
        
        self.criterion_GAN = torch.nn.MSELoss().to(device)
        self.criterion_cycle = torch.nn.L1Loss().to(device)
        self.criterion_identity = torch.nn.L1Loss().to(device)
        
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=learning_rate, betas=(b1, b2)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=learning_rate, betas=(b1, b2))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=learning_rate, betas=(b1, b2))

        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

        self.lambda_id = lambda_id
        self.lambda_cyc = lambda_cyc

        self.Tensor = torch.cuda.FloatTensor if device.type=="cuda" else torch.FloatTensor
        
        self.extra_losses = extra_losses
        self.initWeights()
    
    def train(self, inp, label, compute_extra_losses=True):        
        real_A = inp.type(self.Tensor)
        real_B = label.type(self.Tensor)

        
        valid = self.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape)))
        valid.requires_grad = False
        fake = self.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape)))
        fake.requires_grad = False
        
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()
        
        self.optimizer_G.zero_grad()
        
        # Identity loss
        loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
        loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN & cycle losses
        fake_B = self.G_AB(real_A)
        loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
        loss_cycle_A = self.criterion_cycle(self.G_BA(fake_B), real_A)
        fake_A = self.G_BA(real_B)
        loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
        loss_cycle_B = self.criterion_cycle(self.G_AB(fake_A), real_B)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity

        loss_G.backward()
        self.optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        self.optimizer_D_A.zero_grad()

        # Real loss
        loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_buf = self.fake_A_buffer.push_and_pop(fake_A)
        loss_fake_A = self.criterion_GAN(self.D_A(fake_A_buf.detach()), fake)
        # Total loss
        loss_D_A = (loss_real_A + loss_fake_A) / 2

        loss_D_A.backward()
        self.optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        self.optimizer_D_B.zero_grad()

        # Real loss
        loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B)
        loss_fake_B = self.criterion_GAN(self.D_B(fake_B_buf.detach()), fake)
        # Total loss
        loss_D_B = (loss_real_B + loss_fake_B) / 2

        loss_D_B.backward()
        self.optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        losses = {}
        
        losses["GLoss_id_A"] = loss_id_A.item()
        losses["GLoss_id_B"] = loss_id_B.item()
        losses["GLoss_id"] = loss_identity.item()
        losses["GLoss_adv_AB"] = loss_GAN_AB.item()
        losses["GLoss_adv_BA"] = loss_GAN_BA.item()
        losses["GLoss_adv"] = loss_GAN.item()
        losses["GLoss_cyc_A"] = loss_cycle_A.item()
        losses["GLoss_cyc_B"] = loss_cycle_B.item()
        losses["GLoss_cyc"] = loss_cycle.item()
        losses["GLoss"] = loss_G.item()
        losses["DLoss_real_A"] = loss_real_A.item()
        losses["DLoss_fake_A"] = loss_fake_A.item()
        losses["DLoss_A"] = loss_D_A.item()
        losses["DLoss_real_B"] = loss_real_B.item()
        losses["DLoss_fake_B"] = loss_fake_B.item()
        losses["DLoss_B"] = loss_D_B.item()
        losses["DLoss"] = loss_D.item()

        if compute_extra_losses:
            for k, l in self.extra_losses.items():
                losses[k] = l(fake_B, real_B).item()
        
        return losses
                
    def evaluate(self, inp, label, compute_extra_losses=True):
        self.G_AB.eval()
        self.G_BA.eval()
        self.D_A.eval()
        self.D_B.eval()
        
        with torch.no_grad():
            real_A = inp.type(self.Tensor)
            real_B = label.type(self.Tensor)
    
            valid = self.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape)))
            fake = self.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape)))
      
            loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
            loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)
    
            loss_identity = (loss_id_A + loss_id_B) / 2
    
            fake_B = self.G_AB(real_A)
            loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
            loss_cycle_A = self.criterion_cycle(self.G_BA(fake_B), real_A)
            fake_A = self.G_BA(real_B)
            loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)
            loss_cycle_B = self.criterion_cycle(self.G_AB(fake_A), real_B)
    
            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2
    
            loss_G = loss_GAN + self.lambda_cyc * loss_cycle + self.lambda_id * loss_identity
    

            loss_real_A = self.criterion_GAN(self.D_A(real_A), valid)
            fake_A_buf = self.fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = self.criterion_GAN(self.D_A(fake_A_buf.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
    

            loss_real_B = self.criterion_GAN(self.D_B(real_B), valid)
            fake_B_buf = self.fake_B_buffer.push_and_pop(fake_B)
            loss_fake_B = self.criterion_GAN(self.D_B(fake_B_buf.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
    
            loss_D = (loss_D_A + loss_D_B) / 2
    
            losses = {}
            
            losses["GLoss_id_A"] = loss_id_A.item()
            losses["GLoss_id_B"] = loss_id_B.item()
            losses["GLoss_id"] = loss_identity.item()
            losses["GLoss_adv_AB"] = loss_GAN_AB.item()
            losses["GLoss_adv_BA"] = loss_GAN_BA.item()
            losses["GLoss_adv"] = loss_GAN.item()
            losses["GLoss_cyc_A"] = loss_cycle_A.item()
            losses["GLoss_cyc_B"] = loss_cycle_B.item()
            losses["GLoss_cyc"] = loss_cycle.item()
            losses["GLoss"] = loss_G.item()
            losses["DLoss_real_A"] = loss_real_A.item()
            losses["DLoss_fake_A"] = loss_fake_A.item()
            losses["DLoss_A"] = loss_D_A.item()
            losses["DLoss_real_B"] = loss_real_B.item()
            losses["DLoss_fake_B"] = loss_fake_B.item()
            losses["DLoss_B"] = loss_D_B.item()
            losses["DLoss"] = loss_D.item()
    
            if compute_extra_losses:
                for k, l in self.extra_losses.items():
                    losses[k] = l(fake_B, real_B).item()
            
            return fake_B, losses
    
    def saveToFile(self, filename_stub):
        torch.save(self.G_AB.state_dict(), filename_stub + "_generator_AB.ts")
        torch.save(self.G_BA.state_dict(), filename_stub + "_generator_BA.ts")
        torch.save(self.D_A.state_dict(), filename_stub + "_discriminator_A.ts")
        torch.save(self.D_B.state_dict(), filename_stub + "_discriminator_B.ts")
    
    def loadFromFile(self, filename_stub):
        self.G_AB.load_state_dict(torch.load(filename_stub + "_generator_AB.ts"))
        self.G_BA.load_state_dict(torch.load(filename_stub + "_generator_BA.ts"))
        self.D_A.load_state_dict(torch.load(filename_stub + "_discriminator_A.ts"))
        self.D_B.load_state_dict(torch.load(filename_stub + "_discriminator_B.ts"))
        
    def initWeights(self):
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)
        
    def __call__(self, inp):
        self.G_AB.eval()
        with torch.no_grad():
            return self.G_AB(inp.type(self.Tensor))