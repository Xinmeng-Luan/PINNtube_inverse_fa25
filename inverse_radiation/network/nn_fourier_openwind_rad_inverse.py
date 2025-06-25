"""
Neural network architecture
----------------------------------------------------------------------------
- Acoustic Field Reconstruction in Tubes via Physics-Informed Neural Networks
- Forum Acusticum 25
- by Xinmeng Luan
- xinmeng.luan@mail.mcgill.ca
"""

import torch.nn as nn
import os
os.environ['CUDA_ALLOW_GROWTH'] = 'True'
import inverse_radiation.data.para as PARA
import torch
import rff
import numpy as np

np.random.seed(PARA.seed)
torch.manual_seed(PARA.seed)
torch.cuda.manual_seed(PARA.seed)
torch.cuda.manual_seed_all(PARA.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(PARA.seed)
torch.use_deterministic_algorithms(True)

class Snake(torch.nn.Module):
    """
       Snake activation function
    """
    def forward(self, a):
        out = a + (torch.sin(a))**2
        return out

class FCBlock(nn.Module):

    def __init__(self, nf):
        super(FCBlock, self).__init__()
        self.fc1 = nn.Linear(nf,nf)
        self.af = Snake()
        self.fc2 = nn.Linear(nf, nf)
        self.fc3 = nn.Linear(nf, nf)

    def forward(self, x):
        y = self.fc1(x)
        y = self.af(y)
        y = self.fc2(y)
        y = self.af(y)
        y = self.fc3(y)
        y = x + y
        y = self.af(y)
        return y

class WaveNN(nn.Module):
    def __init__(self, nf, nb):
        super(WaveNN, self).__init__()
        self.af = Snake()
        self.fci = nn.Linear(200, nf)
        self.fcBlocks = nn.ModuleList([FCBlock(nf) for _ in range(nb)])
        self.fco = nn.Linear(nf, 1)

    def forward(self, x, alpha_phi):
        self.af = Snake()
        y = self.fci(x)
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
        self.rad_alpha = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.rad_beta = nn.Parameter(torch.tensor( 1, dtype=torch.float32))

    def forward(self, x_tot, t_tot, alpha_phi):

        x_tot = self.fourier_encode_wavex(x_tot.unsqueeze(1))
        t_tot = self.fourier_encode_wavet(t_tot.unsqueeze(1))
        xt = torch.cat((x_tot, t_tot), dim=1)
        phi = self.wavenn(xt, alpha_phi)

        return phi


