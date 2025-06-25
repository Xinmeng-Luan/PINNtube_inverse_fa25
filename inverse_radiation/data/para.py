"""
Physical parameters
----------------------------------------------------------------------------
- Acoustic Field Reconstruction in Tubes via Physics-Informed Neural Networks
- Forum Acusticum 25
- by Xinmeng Luan
- xinmeng.luan@mail.mcgill.ca
"""

import numpy as np
import scipy.io

seed = 9876
np.random.seed(seed)

# geometry parameters
# constant cross-section case
l = 1 #[m] tube length  # print(geom["l"])
d = 0.01 #[m] tube diameter
S = np.pi *d #[m] circumference
A = np.pi * (d/2)**2

f = 261.6
omega_c = 2*np.pi*f
T = 1/f # [s] -> f = 261.6 Hz (C4)

# physical parameters
rho = 1.20 #[kg/m^3] air density
K = 1.39e5 #[Pa] Bulk modulus
c = 340 #[m/s] speed of sound
mu = 19.0e-6 #[Pa s] viscosity coefficient
eta = 1.40 #heat capacity ratio
lamda_th = 2.41e-2 #[W/(m K)] thermal conductivity
c_p = 1.01e3 #[kJ/(kg K)] specific heat for const. pressure

# viscous and thermal loss
R = S / (A**2) * np.sqrt(omega_c * rho * mu/2) # coefficient of energy loss owing to viscous friction at the tube wall
G = S * (eta-1)/(rho*(c**2)) * np.sqrt(lamda_th * omega_c / (2*c_p*rho)) * np.sqrt(1000) # coefficient of energy loss owing to thermal conduction at the tube wall

# radiation: infinite planar baffle
Rr = 128*rho*c / (9*(np.pi**2)*A) # resistance
Lr = 8*rho / (3*np.pi*np.sqrt(np.pi*A))# reactance


# PDE loss data
NE = 5000
NE_batch = 1000

# BC loss [0,T]
NB = 1000
ti_bc = np.linspace(0, T, num=NB)
# coupling loss [0,T]
NC = 1000
ti_coup = np.linspace(0, T, num=NC)
# periodicity loss [0,l]
NP = 1000
xi_per = np.linspace(0,l,NP)

def vis_points(xi_pde, ti_pde, ti_bc, xi_per):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(xi_pde, ti_pde, c='blue', marker='o', s=10, label='pde points')
    plt.scatter(np.ones(np.shape(ti_bc))*l, ti_bc, c='red', marker='o', s=10, label='coupling points')
    plt.scatter(np.zeros(np.shape(ti_bc)), ti_bc, c='yellow', marker='o', s=10, label='boundary points')
    plt.scatter(xi_per, np.ones(np.shape(xi_per))*T, c='green', marker='o', s=10, label='periodicity points')
    plt.scatter(xi_per, np.zeros(np.shape(xi_per)), c='green', marker='o', s=10, label='periodicity points')

    plt.xlabel('x[m]', fontsize=12)
    plt.ylabel('t[s]', fontsize=12)
    plt.title('Scatter Plot of Points', fontsize=14)

    plt.grid(True)
    plt.legend()
    plt.show()

# TODO: Needed to be changed to your path
data_path = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/smoothedRosenbergWave.mat'
u0 = scipy.io.loadmat(data_path)
u0 = u0['smoothedRosenbergWave']
u0 = u0.squeeze()
u0 = np.append(u0, 0)


def vis_glottal(ti_bc, u0):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(ti_bc, u0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Glottal pulse at x=0 (bc)', fontsize=14)
    plt.grid(True)
    plt.show()

# nn output gain
alpha_phi = 0.006
alpha_u = 1e-6

alpha_uin = 7e-5
alpha_p = 240
alpha_y = 0.41e-3


# NN parameters
Nf = 200 # output nn channels of Input FC layer
Nb = 5 # FC blocks number
epoch = 40002 #TODO
lr = 0.001
beta = 0.007 #0.02

index_pde = range(0, NE)
index_bc = range(NE, NE + NB)
index_coup = range(NE + NB, NE + NB + NC)
index_per1 = range(NE + NB + NC, NE + NB + NC + NP)
index_per2 = range(NE + NB + NC + NP, NE + NB + NC + 2*NP)

index_pde_batch = range(0, NE_batch)
index_bc_batch = range(NE_batch, NE_batch + NB)
index_coup_batch = range(NE_batch + NB, NE_batch + NB + NC)
index_per1_batch = range(NE_batch + NB + NC, NE_batch + NB + NC + NP)
index_per2_batch = range(NE_batch + NB + NC + NP, NE_batch + NB + NC + 2*NP)

lambda_B = 3.4e5
lambda_u = 5e4
lambda_l = 5e6
lambda_r = 5e6
lambda_E = 5e-6
lambda_p = 1
lambda_t = 1e-8
lambda_P = 1
lambda_C = 1

lambda_pt = 1.8e-5
lambda_px = 0.0023
lambda_ut = 4.3e8
lambda_ux = 3e10
lambda_Eu = 3e10*100000
lambda_Ep = 0.0023 *100000

lambda_Id = 58
lambda_Iv = 11

lambda_rad = 0.01

N_ff = 512 # fourier feature encoded size
f_gradnorm = 1000 # gradnorm update period
snake_alpha = 0.5 #TODO
para_variance = 1+ (1+np.exp(-8*(snake_alpha**2)) - 2* np.exp(-4*(snake_alpha**2)))/(8*(snake_alpha**2))

loss_num = 7
lr_2 = 1e-2
gradnorm_per_epoch = 100

cx = l/2
ct = T/2
ct_tran = 1*T/2


## radiation (openwind), infinite_flanged
# "Time-domain simulation of a dissipative reed instrument (2020) Thibault"
# taylor (chaigne)
rad_delta = 0.8236
rad_beta_chaigne = 0.5
# pade
rad_alpha = 1 / rad_delta # 1.2141816415735793
rad_beta = rad_beta_chaigne / (rad_delta ** 2) # 0.737118529367156
def constant_with_initial_ramp(pm_max=5000, t_car=5e-2, t_down=1e10):

    def pm(t):
        if np.isscalar(t):  # For scalars
            if 0 < t < t_car:  # Lighter computation for scalars
                return pm_max * (1 - np.cos(np.pi * t / t_car)) / 2
            elif 0 < t - t_down < t_car:
                return pm_max * (1 + np.cos(np.pi * (t - t_down) / t_car)) / 2
            elif t_car <= t <= t_down:
                return pm_max
            else:
                return 0
        ramp = 1.0 * (0 < t) * (t < t_car) * (1 - np.cos(np.pi * t / t_car)) / 2
        ramp2 = ((0 < t - t_down) * (t - t_down < t_car) *
                 (1 + np.cos(np.pi * (t - t_down) / t_car)) / 2)
        return pm_max * (ramp + (t_car <= t) * (t <= t_down) + ramp2)

    return pm


batch_size = 10,000