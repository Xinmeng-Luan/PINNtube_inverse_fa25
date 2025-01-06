"""
Neural network architecture
"""


import torch
import torch.nn as nn
import sys
import os
os.environ["LD_LIBRARY_PATH"] = "/nas/home/xluan/miniconda3/envs/thesis/lib:/nas/home/xluan/miniconda3/envs/thesis/lib/cuda/lib64"
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
import PINN_wind.pinn_wind.main.data.para as PARA
from torch.nn import functional as F
# Nf = PARA.Nf
import torch
# from snake.activations import Snake
import rff
import numpy as np

# set seed of random numpy
np.random.seed(PARA.seed)
torch.manual_seed(PARA.seed)
torch.cuda.manual_seed(PARA.seed)
torch.cuda.manual_seed_all(PARA.seed)
# When running on the CuDNN backend, two further options must be set
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Set a fixed value for the hash seed
os.environ["PYTHONHASHSEED"] = str(PARA.seed)
torch.use_deterministic_algorithms(True)



class Snake(torch.nn.Module):
    """
       Snake activation function
    """
    def forward(self, a):
        out = a + (torch.sin(a))**2 #TODO
        return out

class FCBlock(nn.Module):

    def __init__(self, nf):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(nf,nf)
        self.af = Snake()
        # self.af1 = Snake(nf, PARA.snake_alpha)
        self.fc2 = nn.Linear(nf, nf)
        # self.af2 = Snake(nf, PARA.snake_alpha)
        self.fc3 = nn.Linear(nf, nf)
        # self.af3 = Snake(nf, PARA.snake_alpha)

    def forward(self, x):
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=PARA.para_variance ** 0.5)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=PARA.para_variance ** 0.5)
        # nn.init.normal_(self.fc3.weight, mean=0.0, std=PARA.para_variance ** 0.5)
        y = self.fc1(x)
        # y = self.af1(y)
        y = self.af(y)
        y = self.fc2(y)
        # y = self.af2(y)
        y = self.af(y)
        y = self.fc3(y)
        y = x + y
        # y = self.af3(y)
        y = self.af(y)
        return y

class WaveNN(nn.Module):
    def __init__(self, nf, nb):
        super(WaveNN, self).__init__()
        self.af = Snake()
        self.fci = nn.Linear(200, nf) #TODO 2
        # self.af = Snake(nf, PARA.snake_alpha)
        self.fcBlocks = nn.ModuleList([FCBlock(nf) for _ in range(nb)])
        self.fco = nn.Linear(nf, 1)

    def forward(self, x, alpha_phi):
        # nn.init.normal_(self.fci.weight, mean=0.0, std=PARA.para_variance ** 0.5)
        # nn.init.normal_(self.fco.weight, mean=0.0, std=PARA.para_variance ** 0.5)
        # import matplotlib.pyplot as plt
        # plt.plot(layer.weight.detach().cpu().numpy())
        # plt.show()
        #
        # plt.plot(layer.weight.detach().cpu().numpy())
        # plt.show()
        self.af = Snake()
        y = self.fci(x)
        # y = self.af(y)
        y = self.af(y)
        for fcBlock in self.fcBlocks:
            y = fcBlock(y)
        y = self.fco(y)
        y = y * alpha_phi
        return y

class MainNN(nn.Module):
    def __init__(self, nf, nb):
        super(MainNN, self).__init__()
        self.wavenn = WaveNN(nf, nb)
        self.fourier_encode_wavex = rff.layers.GaussianEncoding(sigma=0.1, input_size=1, encoded_size=int(50))
        self.fourier_encode_wavet = rff.layers.GaussianEncoding(sigma=0.1, input_size=1, encoded_size=int(50))

        # self.rad_alpha = nn.Parameter(torch.tensor(1.036750, dtype=torch.float32))
        # self.rad_beta = nn.Parameter(torch.tensor(0.739557, dtype=torch.float32))
        self.rad_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.rad_beta = nn.Parameter(torch.tensor( 1, dtype=torch.float32))


        # self.rad_delta = nn.Parameter(torch.tensor(0.6133, dtype=torch.float32))
        # self.rad_beta_chaigne = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
    def forward(self, x_tot, t_tot, alpha_phi):

        x_tot = self.fourier_encode_wavex(x_tot.unsqueeze(1))
        t_tot = self.fourier_encode_wavet(t_tot.unsqueeze(1))
        xt = torch.cat((x_tot, t_tot), dim=1)
        phi = self.wavenn(xt, alpha_phi)

        return phi


