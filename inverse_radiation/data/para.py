"""
Physical parameters
----------------------------------------------------------------------------
- Acoustic Field Reconstruction in Tubes via Physics-Informed Neural Networks
- Forum Acusticum 25
- by Xinmeng Luan
- xinmeng.luan@mail.mcgill.ca
"""

import numpy as np
import sys
import scipy.io

seed = 9876
np.random.seed(seed)


# geometry parameters
# constant cross-section case
l = 1 #[m] tube length  # print(geom["l"])
d = 0.01 #[m] tube diameter
S = np.pi *d #[m] circumference
A = np.pi * (d/2)**2


# f = 440*2**(-9/12) # [Hz] #261.6
f = 261.6
omega_c = 2*np.pi*f
T = 1/f # [s] -> f = 261.6 Hz (C4)

# physical parameters
rho = 1.20 #[kg/m^3] air density
K = 1.39e5 #[Pa] Bulk modulus
c = 340 #[m/s] speed of sound
mu = 19.0e-6 #[Pa s] viscosity coefficient
eta = 1.40 # heat capacity ratio
lamda_th = 2.41e-2 #[W/(m K)] thermal conductivity
c_p = 1.01e3# TODO [kJ/(kg K)] specific heat for const. pressure

# viscous and thermal loss
R = S / (A**2) * np.sqrt(omega_c * rho * mu/2) # coefficient of energy loss owing to viscous friction at the tube wall
G = S * (eta-1)/(rho*(c**2)) * np.sqrt(lamda_th * omega_c / (2*c_p*rho)) * np.sqrt(1000) #TODO * np.sqrt(1000)  # coefficient of energy loss owing to thermal conduction at the tube wall

# radiation: infinite planar baffle
Rr = 128*rho*c / (9*(np.pi**2)*A) # resistance
Lr = 8*rho / (3*np.pi*np.sqrt(np.pi*A))# reactance


# PDE loss data TODO
NE = 5000 #ori:5000
NE_batch = 1000

# samples not change along epochs:
# sobol = Sobol(d=2, scramble=True)
# samples = sobol.random(n=2**13)
# # no normalized
# xi_pde = samples[:NE, 0] * l # [0,l]
# ti_pde = samples[:NE, 1] * T # [0,T]



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


# vis_points(xi_pde, ti_pde, ti_bc, xi_per)

# BC: excitation at x=0 (ground truth)
# glottal pulse [ROSENBERG model],
# implement from eq. 3 in "Stochastic models of glottal pulses from the Rosenberg and Liljencrants-Fant models with unified parameters"
# ori paper: T_p = 40, T_N = 16, T_tot = 60, A_v= 1000

# T_p = 40/60*T
# T_N = 16/60*T
# A_v = 1 # amplitude
#
# N_t1 = round(40/60 * NB) #667
# N_t2 = round(16/60 * NB) #267
# N_t3 = round(4/60 * NB) -1  # 66
# assert N_t1 + N_t2 + N_t3 == NB, "Glottal pulse sampling error!"
# index_t1 = range(0, N_t1)
# index_t2 = range(N_t1, N_t1 + N_t2)
# index_t3 = range(N_t1 + N_t2, N_t1 + N_t2 + N_t3)
#
# u01 = 0.5 * A_v * (1-np.cos(np.pi*ti_bc[index_t1]/(T_p)))
# u02 = A_v * np.cos(np.pi * (ti_bc[index_t2]-T_p)/(2*T_N))
# u03 = 0 * ti_bc[index_t3]
# # u0 = A* np.concatenate((u01, u02, u03), axis=0) # volume velocity
# u0 = np.concatenate((u01, u02, u03), axis=0) # velocity

# velocity
data_path = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/smoothedRosenbergWave.mat'
u0 = scipy.io.loadmat(data_path)
u0 = u0['smoothedRosenbergWave']
u0 = u0.squeeze()
u0 = np.append(u0, 0)

# data_path = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/sin_p.mat'
# u0 = scipy.io.loadmat(data_path)
# u0 = u0['sin_p']
# u0 = u0.squeeze()


# window_size = 5  # You can adjust this size based on your needs
# u0_av = np.convolve(u0, np.ones(window_size)/window_size, mode='valid')


# def moving_average(u0, window_size):
#     return np.convolve(u0, np.ones(window_size) / window_size, mode='valid')
#
# window_size = 10
# u0_smoothed = moving_average(u0, window_size)
#
# print("Original u0:", u0)
# print("Smoothed u0:", u0_smoothed)
# import matplotlib.pyplot as plt
# plt.plot(u0_smoothed)
# plt.show()

def vis_glottal(ti_bc, u0):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(ti_bc, u0)
    plt.xlabel('t', fontsize=12)
    plt.ylabel('u', fontsize=12)
    plt.title('Glottal pulse at x=0 (bc)', fontsize=14)
    plt.grid(True)
    plt.show()

# vis_glottal(ti_bc, u0)
# nn output gain
alpha_phi = 0.006
alpha_u = 1e-6

alpha_uin = 7e-5 # -4.0017e-06, 7.8540e-05
alpha_p = 240
alpha_y = 0.41e-3  #TODO y_eq


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



# loss weight
    # lambda_Bp, lambda_up, lambda_lp, lambda_rp: close order
    # (since the order of the particle velocity u is nearly constant in the tube)
# lambda_Bp = 1 # normalized B' # for the known boundary condition at x = 0, the maximum particle velocity is 1 m/s
lambda_B = 3.4e5
# lambda_up = 1 # empirically
lambda_u = 5e4 #TODO:check :u or v, v:5e4
# lambda_lp = 1 # empirically
lambda_l = 5e6
# lambda_rp = 50 # empirically
lambda_r = 5e6
# lambda_E, lambda_p: ~ p, phi (cannot infer order from the BCs)
lambda_E = 5e-6 # empirically #TODO???? ori:0.58
lambda_p = 1 # empirically
lambda_t = 1e-8 # empirically
# others?
lambda_P = 1
lambda_C = 1 #TODO

lambda_pt = 1.8e-5
lambda_px = 0.0023
lambda_ut = 4.3e8 #u
lambda_ux = 3e10 #u
lambda_Eu = 3e10*100000
lambda_Ep = 0.0023 *100000 # 30

lambda_Id = 58
lambda_Iv = 11

lambda_rad = 0.01 # ori: 0.001

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


## Reed parameter
# initial case from 'Investigating Clarinet Articulation Using a Physical Model andanArtificial Blowing Machine (Vasileios)'

def constant_with_initial_ramp(pm_max=5000, t_car=5e-2, t_down=1e10):
    """Curve starting at zero and reaching a constant mouth pressure.

    Parameters
    ----------
    pm_max : float
        Maximal value
    t_car : float
        Time to reach the value
    t_down : float
        Time when pressure source turns off
    From: openwind: temporal_curves.py
    """
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

# pm = constant_with_initial_ramp(2000, 2e-2)
# pressure = np.array([pm(t) for t in ti_bc])

reed = {
    'L': 0.3294/0.2468, # [m]
    'm': 3.25e-6, # [kg]
    'k': 1765, # [m]
    'gamma': 3000, # [s^-1]
    'y_lay': 0.246e-3, # [m]
    'y_eq': 0.41e-3, # [m]
    'k_lay': 1.97e6, # [N/m^2]
    'r_lay': 2, # [s/m]
    'k_tg': 3.16e4, # [N/m^alpha_tg]
    'r_tg': 2, # [s/m]
    'y_tg_max': 0.51e-3, # [m]
    't1_c': 0.335/0.424, # [s]
    't2_c': 0.34/0.429, # [s]
    't3_c': 0.39/0.53, # [s]
    't4_c': 0.45/0.56, # [s]
    'p_b': 2000, #TODO: constant_with_initial_ramp(2000, 2e-2)
    'alpha_tg': 1.2,
    'alpha_lay': 2,
    'lambda': 1.2e-2, # reed width [m]
    'Sr': 98.6e-6 # effective reed area [m^2]
    }


batch_size = 10,000