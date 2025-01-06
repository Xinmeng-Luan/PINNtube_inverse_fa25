from scipy.io import loadmat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import PINN_wind.pinn_wind.main.data.para as PARA
print('start...')
obser_data_path_end = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/fd_p_all.mat'
data_end = loadmat(obser_data_path_end)
p_all_fd = data_end['pFDM_all'].squeeze()
x_all_fd = data_end['x'].squeeze()
t_all_fd = data_end['t'].squeeze()


file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_all_predict_noise_snr_40_lbfgs.pkl'
with open(file_name, 'rb') as file:
    loaded_data = pickle.load(file)

p_all_predict_noise= loaded_data['p_fd']
p_all_predict_noise = p_all_predict_noise.T
loss = np.abs(p_all_predict_noise - p_all_fd)

# Create a figure and subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot p_all_predict_noise
im1 = axes[0].imshow(p_all_predict_noise, aspect='auto', origin='lower', cmap='jet', extent=[0, PARA.l, 0, PARA.T*1e3])
axes[0].set_title(r'PINN ($\Gamma$)', fontsize=25)
axes[0].set_xlabel(r'$x$ [m]' , fontsize=25)
axes[0].set_ylabel(r'$t$ [ms]', fontsize=25)
cbar0 = fig.colorbar(im1, ax=axes[0], label=r'$p$ [Pa]')
cbar0.set_label(r'$p$ [Pa]', fontsize=25)
cbar0.ax.tick_params(labelsize=20)
axes[0].tick_params(axis='x', labelsize=20)
axes[0].tick_params(axis='y', labelsize=20)


# Plot p_all_fd
im2 = axes[1].imshow(p_all_fd, aspect='auto', origin='lower', cmap='jet', extent=[0, PARA.l, 0, PARA.T*1e3])
axes[1].set_title(r'FDM', fontsize=25)
axes[1].set_xlabel(r'$x$ [m]', fontsize=25)
# axes[1].set_ylabel(r'$x$', fontsize=15)
cbar1 = fig.colorbar(im2, ax=axes[1], label=r'$p$ [Pa]')
cbar1.set_label(r'$p$ [Pa]', fontsize=25)
cbar1.ax.tick_params(labelsize=20)
axes[1].tick_params(axis='x', labelsize=20)
axes[1].tick_params(axis='y', labelsize=20)


# Plot loss
im3 = axes[2].imshow(loss, aspect='auto', origin='lower', cmap='jet', extent=[0, PARA.l, 0, PARA.T*1e3])
axes[2].set_title(r'Error', fontsize=25)
axes[2].set_xlabel(r'$x$ [m]', fontsize=25)
# axes[2].set_ylabel(r'$x$', fontsize=15)
cbar2 = fig.colorbar(im3, ax=axes[2], label=r'$|\Delta p|$ [Pa]')
cbar2.set_label(r'$|\Delta p|$ [Pa]', fontsize=25)
cbar2.ax.tick_params(labelsize=20)
axes[2].tick_params(axis='x', labelsize=20)
axes[2].tick_params(axis='y', labelsize=20)

# Adjust layout and show
plt.tight_layout()
plt.savefig('/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_all_lbfgs.png', dpi=300)

plt.show()


print('')
