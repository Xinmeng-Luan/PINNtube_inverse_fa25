from scipy.io import loadmat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import PINN_wind.pinn_wind.main.data.para as PARA
from scipy.io import loadmat

data_lbfgs = loadmat(file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/radiation_for_predict_end_noise_snr_40_lbfgs.mat')
data_fd = loadmat(file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/radiation_for_predict_fd.mat')


# Access a specific variable (replace 'variable_name' with the actual variable name)
p_lbfgs = data_lbfgs['p_end']
u_t_lbfgs = data_lbfgs['u_t_end']
p_t_lbfgs = data_lbfgs['p_t_end']
p_fd = data_fd['p_end']
u_t_fd = data_fd['u_t_end']
p_t_fd = data_fd['p_t_end']



# obser_data_path_end = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/observe_end_infinite_flanged_data.mat'
# data_end = loadmat(obser_data_path_end)
# p_end_infinite_flanged_gt = data_end['p_end_infinite_flanged'].squeeze()

signal_power = np.mean(p_fd.squeeze() ** 2)
SNR_dB = 40
# Calculate the noise power from the SNR
SNR_linear = 10 ** (SNR_dB / 10)  # Convert SNR from dB to linear scale
noise_power = signal_power / SNR_linear
# Generate AWGN noise with zero mean and the calculated noise power
noise = np.sqrt(noise_power) * np.random.randn(len(p_fd.squeeze()))
p_end_infinite_flanged_noise_gt = p_fd.squeeze() + noise

# file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_predict_noise_snr_40.pkl'
# with open(file_name, 'rb') as file:
#     loaded_data = pickle.load(file)
# t_end = loaded_data['t_coup'].detach().cpu().numpy()
# p_end_infinite_flanged_noise_predict = loaded_data['p_coup'].detach().cpu().numpy()

# file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_predict.pkl'
# with open(file_name, 'rb') as file:
#     loaded_data = pickle.load(file)
# p_end_infinite_flanged_predict = loaded_data['p_coup'].detach().cpu().numpy()


t_end_fd  = np.linspace(-1,1,1001)
t_end_fd = (t_end_fd + 1)/2*PARA.T

t_end_lbfgs  = np.linspace(-1,1,1000)
t_end_lbfgs  = (t_end_lbfgs  + 1)/2*PARA.T


fig, axes = plt.subplots(1, 3, figsize=(18, 4))  # Adjust the figure size as needed

# First subplot (1, 3, 1)
axes[0].plot(t_end_fd.squeeze(), p_fd.squeeze(), label='FDM', color='black', linestyle='-', linewidth=2, alpha=1)
axes[0].plot(t_end_fd.squeeze(), p_end_infinite_flanged_noise_gt.squeeze(), label='Noisy FDM', color='blue', linestyle='-', linewidth=2, alpha=0.5)
axes[0].plot(t_end_lbfgs.squeeze(), p_lbfgs.squeeze(), label='PINN', color='red', linestyle=':', linewidth=2, alpha=1)
axes[0].set_xlabel(r'$t$ [s]', fontsize=20)
axes[0].set_ylabel(r'$p$ [Pa]', fontsize=20)
axes[0].legend(loc='upper right', fontsize=15)
axes[0].tick_params(axis='both', labelsize=18)
# axes[0].set_title("Pressure Comparison", fontsize=20)

# Second subplot (1, 3, 2)
axes[1].plot(t_end_fd.squeeze(), p_t_fd.squeeze(), label='FDM', color='black', linestyle='-', linewidth=2, alpha=1)
axes[1].plot(t_end_lbfgs.squeeze(), p_t_lbfgs.squeeze(), label='PINN', color='red', linestyle=':', linewidth=2, alpha=1)
axes[1].set_xlabel(r'$t$ [s]', fontsize=20)
axes[1].set_ylabel(r'$p_t$ [Pa/s]', fontsize=20)
axes[1].legend(loc='lower right', fontsize=15)
axes[1].tick_params(axis='both', labelsize=18)
# axes[1].set_title("Pressure Derivative", fontsize=20)

# Third subplot (1, 3, 3)
axes[2].plot(t_end_fd.squeeze(), u_t_fd.squeeze(), label='FDM', color='black', linestyle='-', linewidth=2, alpha=1)
axes[2].plot(t_end_lbfgs.squeeze(), u_t_lbfgs.squeeze(), label='PINN', color='red', linestyle=':', linewidth=2, alpha=1)
axes[2].set_xlabel(r'$t$ [s]', fontsize=20)
axes[2].set_ylabel(r'$u_t$ [m/s]', fontsize=20)
axes[2].legend(loc='lower right', fontsize=15)
axes[2].tick_params(axis='both', labelsize=18)
# axes[2].set_title("Velocity Derivative", fontsize=20)

plt.tight_layout()

# Save the figure
plt.savefig('/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_all_subplots.png', dpi=300)

plt.show()

print('')