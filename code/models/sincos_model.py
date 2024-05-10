import math
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SinCosHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SinCosHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).to(DEVICE).float() + 0.5)

    def forward(self, x):
        x = torch.tanh(x)
        x_cos = 1 - (torch.cos(0.5 * math.pi * x) ** 2)  # side bin
        x_sin = 1 - (torch.sin(0.5 * math.pi * x) ** 2)  # center bin
        x = torch.stack([
            torch.sum(x_cos * (x <= 0)),  # Where tanh(D(x)) is counted as -1
            torch.sum(x_sin),             # Where tanh(D(x)) is counted as 0
            torch.sum(x_cos * (x > 0))    # Where tanh(D(x)) is counted as 1
        ])
        # Final output size is [3, 1], when input size is [batch_size]
        # The input size may be not [batch_size, 1], because of squeeze operation
        return x.unsqueeze(dim=1)

# 0
# WGAN

# 1
# LSGAN

# 2
class Euclidean(nn.Module):
    def __init__(self, batch_size, n_bin):
        super(Euclidean, self).__init__()
        self.batch_size = batch_size
        self.n_bin = n_bin

    def forward(self, x, y):
        gausshist = SinCosHistogram(bins=self.n_bin, min=0, max=1, sigma=0.5)
        x, y = torch.squeeze(x), torch.squeeze(y)
        x_hist, y_hist = gausshist(x)/self.batch_size, gausshist(y)/self.batch_size
        diff_squared = torch.pow((x_hist - y_hist), 2)
        euclidean_distance = torch.sqrt(torch.sum(diff_squared, dim=0))
        return euclidean_distance

# 3
class Chi2(nn.Module):
    def __init__(self, batch_size, n_bin):
        super(Chi2, self).__init__()
        self.batch_size = batch_size
        self.n_bin = n_bin

    def forward(self, x, y):
        gausshist = SinCosHistogram(bins=self.n_bin, min=0, max=1, sigma=0.5)
        x, y = torch.squeeze(x), torch.squeeze(y)
        x_hist, y_hist = gausshist(x)/self.batch_size, gausshist(y)/self.batch_size
        epsilon = 1e-10
        diff_squared = torch.pow(x_hist - y_hist, 2)
        chi_squared_distance = torch.sum(diff_squared / (x_hist + y_hist + epsilon), dim=0)
        return chi_squared_distance

# 4
class Chybyshev(nn.Module):
    def __init__(self, batch_size, n_bin):
        super(Chybyshev, self).__init__()
        self.batch_size = batch_size
        self.n_bin = n_bin

    def forward(self, x, y):
        gausshist = SinCosHistogram(bins=self.n_bin, min=0, max=1, sigma=0.5)
        x, y = torch.squeeze(x), torch.squeeze(y)
        x_hist, y_hist = gausshist(x)/self.batch_size, gausshist(y)/self.batch_size
        chybyshev_distance = torch.max(torch.abs(x_hist - y_hist), dim=0)[0]
        return chybyshev_distance

# 5
class Manhattan(nn.Module):
    def __init__(self, batch_size, n_bin):
        super(Manhattan, self).__init__()
        self.batch_size = batch_size
        self.n_bin = n_bin

    def forward(self, x, y):
        gausshist = SinCosHistogram(bins=self.n_bin, min=0, max=1, sigma=0.5)
        x, y = torch.squeeze(x), torch.squeeze(y)
        x_hist, y_hist = gausshist(x)/self.batch_size, gausshist(y)/self.batch_size
        manhattan_distance = torch.sum(torch.abs(x_hist - y_hist), dim=0)
        return manhattan_distance

# 6
class SquaredChord(nn.Module):
    def __init__(self, batch_size, n_bin):
        super(SquaredChord, self).__init__()
        self.batch_size = batch_size
        self.n_bin = n_bin

    def forward(self, x, y):
        gausshist = SinCosHistogram(bins=self.n_bin, min=0, max=1, sigma=0.5)
        x, y = torch.squeeze(x), torch.squeeze(y)
        x_hist, y_hist = gausshist(x)/self.batch_size, gausshist(y)/self.batch_size
        c = torch.pow((torch.sqrt(x_hist) - torch.sqrt(y_hist)), 2)
        sc_distance = torch.sum(c, dim=0)
        return sc_distance
