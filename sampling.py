from torch.utils.data import Dataset
import os
import numpy as np
import pydicom
from torchvision import transforms as T
from numpy.ma import masked_array
import cv2
from utils import extract

class DiffusionSampler:
    def __init__(self, model, encoder, decoder, num_timesteps):
        self.model = model
        self.encoder = encoder.eval()
        self.decoder = decoder.eval()
        self.num_timesteps = num_timesteps

    def sample(self, img2):
        img2 = self.encoder(img2)
        self.model.eval()
        t = self.num_timesteps
        batch_size = 1
        xt = img2
        direct_recons = None
        imgs = []

        while t:
            input = torch.cat((img2, xt), dim=1)
            step = torch.full((batch_size,), t - 1, dtype=torch.long).cuda()
            x1_bar = self.model(input, step)
            x2_bar = self.get_x2_bar_from_xt2(x1_bar, img2, step)

            if direct_recons is None:
                direct_recons = x1_bar

            xt_bar = x1_bar
            if t != 0:
                xt_bar = self.q_sample(x_start=xt_bar, x_end=x2_bar, t=step)

            xt_sub1_bar = x1_bar
            if t - 1 != 0:
                step2 = torch.full((batch_size,), t - 2, dtype=torch.long).cuda()
                xt_sub1_bar = self.q_sample(x_start=xt_sub1_bar, x_end=x2_bar, t=step2)

            x = img2 - xt_bar + xt_sub1_bar
            img2 = x
            if t == 1:
                output = self.decoder(img2)
                imgs.append(output)
            t -= 1

        self.model.train()
        return imgs[0]

    def get_x2_bar_from_xt2(self, x1_bar, xt, t):
        return (xt - x1_bar) / extract(self.model.betas, t, x1_bar.shape)

    def q_sample(self, x_start, x_end, t, noise=None):
        noise = noise or (x_end - x_start)
        return x_start + extract(self.model.betas, t, x_start.shape) * x_end
