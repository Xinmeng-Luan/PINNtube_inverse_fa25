from scipy.io import loadmat
import pickle
import numpy as np
import matplotlib.pyplot as plt
import PINN_wind.pinn_wind.main.data.para as PARA

obser_data_path_end = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/observe_end_infinite_flanged_data.mat'
data_end = loadmat(obser_data_path_end)
p_end_infinite_flanged_gt = data_end['p_end_infinite_flanged'].squeeze()



signal_power = np.mean(p_end_infinite_flanged_gt ** 2)
SNR_dB = 40
# Calculate the noise power from the SNR
SNR_linear = 10 ** (SNR_dB / 10)  # Convert SNR from dB to linear scale
noise_power = signal_power / SNR_linear
# Generate AWGN noise with zero mean and the calculated noise power
noise = np.sqrt(noise_power) * np.random.randn(len(p_end_infinite_flanged_gt))
p_end_infinite_flanged_noise_gt = p_end_infinite_flanged_gt + noise

file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_predict_noise_snr_40.pkl'
with open(file_name, 'rb') as file:
    loaded_data = pickle.load(file)
t_end = loaded_data['t_coup'].detach().cpu().numpy()
p_end_infinite_flanged_noise_predict = loaded_data['p_coup'].detach().cpu().numpy()


file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_predict.pkl'
with open(file_name, 'rb') as file:
    loaded_data = pickle.load(file)
p_end_infinite_flanged_predict = loaded_data['p_coup'].detach().cpu().numpy()

t_end = (t_end + 1)/2*PARA.T

plt.figure(figsize=(12, 6))
plt.plot( t_end,p_end_infinite_flanged_gt, label='Clean Input', color='black', linestyle='-', linewidth=9 , alpha=0.2)
plt.plot(t_end,p_end_infinite_flanged_noise_gt, label='Noisy Input', color='blue', linestyle='-', linewidth=3, alpha=0.8)
plt.plot( t_end,p_end_infinite_flanged_predict , label='Prediction with Clean Input', color='red',linestyle=':', linewidth=2, alpha=1)
plt.plot( t_end,p_end_infinite_flanged_noise_predict, label='Prediction with Noisy Input', color='green', linestyle='-.', linewidth=2, alpha=1)
# plt.title('Losses Over Epochs', fontsize=14)
plt.xlabel(r'$t$ [s]', fontsize=30)
plt.ylabel(r'$p$ [Pa]', fontsize=30)
plt.legend(loc='upper right', fontsize=20)
plt.xticks(fontsize=20)  # Set the font size of x-axis tick labels
plt.yticks(fontsize=20)
# plt.grid(True)
plt.tight_layout()
plt.savefig('/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_end_all.png', dpi=300)
plt.show()

print('')
