"""
main training script
----------------------------------------------------------------------------
- Acoustic Field Reconstruction in Tubes via Physics-Informed Neural Networks
- Forum Acusticum 25
- by Xinmeng Luan
- xinmeng.luan@mail.mcgill.ca
"""

import os
import sys
import torch
import inverse_radiation.data.para as PARA
from scipy.stats.qmc import Sobol
import pickle
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
import scipy.io

os.environ['CUDA_ALLOW_GROWTH'] = 'True'
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# set seed of random numpy
np.random.seed(PARA.seed)
torch.manual_seed(PARA.seed)
torch.cuda.manual_seed(PARA.seed)
torch.cuda.manual_seed_all(PARA.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(PARA.seed)
torch.use_deterministic_algorithms(True)
print(f"Random seed set as {PARA.seed}")

# Check if GPUs are available
print("Check GPUs----")
try:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}")
        device = torch.device('cuda:0')

    else:
        raise Exception("No GPUs available.")
except Exception as e:
    print(f"GPU Error: {str(e)}")
    sys.exit()

now = datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))



Ai_pde = PARA.A
Ai_x_pde = 0


class Trainer:
    def __init__(self, model, main_optimizer,n_epochs):

        # Storage for losses
        self.tot_losses = []
        self.E_losses = []
        self.B_losses = []
        self.P_losses = []
        self.R_losses = []
        self.O_losses = []
        self.O1_losses = []
        self.rad_alphas = []
        self.rad_betas = []

        self.model = model
        self.main_optimizer = main_optimizer
        self.n_epochs = n_epochs

    def rand_pde_points(self):
        sobol = Sobol(d=2, scramble=True)
        samples = sobol.random(n=2 ** 13)
        xi_pde = samples[:PARA.NE, 0] * PARA.l  # [0,l]
        ti_pde = samples[:PARA.NE, 1] * PARA.T  # [0,T]
        return xi_pde, ti_pde

    def get_data(self):
        #TODO: Needed to be changed to your path
        input_data_path = '/nas/home/xluan/thesis-xinmeng/PINN_wind/input_data/data_input_pinn_forward.pkl'
        with open(input_data_path, 'rb') as f:
            data = pickle.load(f)

        x_pde = data['x_pde'].to(device)
        t_pde = data['t_pde'].to(device)
        x_bc = data['x_bc'].to(device)
        t_bc = data['t_bc'].to(device)
        x_coup = data['x_coup'].to(device)
        t_coup = data['t_coup'].to(device)
        x_per0 = data['x_per0'].to(device)
        t_per0 = data['t_per0'].to(device)
        x_perT = data['x_perT'].to(device)
        t_perT = data['t_perT'].to(device)

        # TODO: Needed to be changed to your path
        obser_data_path_end = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/observe_end_infinite_flanged_data.mat'
        data_end = loadmat(obser_data_path_end)
        p_end_infinite_flanged = data_end['p_end_infinite_flanged'].squeeze()
        # Add noise
        signal_power = np.mean(p_end_infinite_flanged ** 2)
        SNR_dB = 40
        SNR_linear = 10 ** (SNR_dB / 10)  # Convert SNR from dB to linear scale
        noise_power = signal_power / SNR_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(p_end_infinite_flanged))
        p_end_infinite_flanged = p_end_infinite_flanged + noise
        p_end_infinite_flanged = torch.tensor(p_end_infinite_flanged).squeeze().to(device).to(torch.float32)

        x_end_ob = torch.tensor(data_end['x']).squeeze().to(device).to(torch.float32)
        t_end_ob = torch.tensor(data_end['t']).squeeze().to(device).to(torch.float32)

        # normalize
        x_pde = x_pde / PARA.l * 2 - 1
        t_pde = t_pde / PARA.T * 2 - 1
        x_bc = x_bc / PARA.l * 2 - 1
        t_bc = t_bc / PARA.T * 2 - 1
        x_coup = x_coup / PARA.l * 2 - 1
        t_coup = t_coup / PARA.T * 2 - 1
        x_per0 = x_per0 / PARA.l * 2 - 1
        t_per0 = t_per0 / PARA.T * 2 - 1
        x_perT = x_perT / PARA.l * 2 - 1
        t_perT = t_perT / PARA.T * 2 - 1
        x_end_ob = x_end_ob / PARA.l * 2 - 1
        t_end_ob = t_end_ob / PARA.T * 2 - 1

        x_pde.requires_grad = True
        t_pde.requires_grad = True
        x_bc.requires_grad = True
        t_bc.requires_grad = True
        x_coup.requires_grad = True
        t_coup.requires_grad = True
        x_per0.requires_grad = True
        t_per0.requires_grad = True
        x_perT.requires_grad = True
        t_perT.requires_grad = True
        x_end_ob.requires_grad = True
        t_end_ob.requires_grad = True
        p_end_infinite_flanged.requires_grad = True

        return (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
                x_end_ob, t_end_ob, p_end_infinite_flanged)

    def calc_grad(self, y, x) -> torch.Tensor:
        grad = torch.autograd.grad(
            outputs=y,  # todo
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0]
        return grad

    def get_p(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_t = self.calc_grad(phi, t) / PARA.ct
        p = PARA.R * PARA.A * phi.squeeze() + PARA.rho * phi_t

        return  p

    def get_u(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_x = self.calc_grad(phi, x) / PARA.cx
        u = -PARA.A * phi_x

        return  u

    def get_end_for_radiation(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_x = self.calc_grad(phi, x) / PARA.cx
        phi_t = self.calc_grad(phi, t) / PARA.ct
        u = -PARA.A * phi_x
        p = PARA.R * PARA.A * phi.squeeze() + PARA.rho * phi_t
        u_t = self.calc_grad(u, t) / PARA.ct
        p_t = self.calc_grad(p, t) / PARA.ct

        return  u_t.detach().cpu().numpy(), p.detach().cpu().numpy(), p_t.detach().cpu().numpy()


    def pde_loss(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_x = self.calc_grad(phi, x) / PARA.cx
        phi_xx = self.calc_grad(phi_x, x) / PARA.cx
        phi_t = self.calc_grad(phi, t) / PARA.ct
        phi_tt = self.calc_grad(phi_t, t) / PARA.ct
        l_E = torch.mean((PARA.c ** 2 * (phi_xx
                                         + 1 / PARA.A * Ai_x_pde * phi_x
                                         - PARA.G * PARA.R * phi
                                         - (PARA.G * PARA.rho / PARA.A + PARA.R * PARA.A / PARA.K) * phi_t
                                         - PARA.rho / PARA.K * phi_tt)) ** 2)
        p = PARA.R * PARA.A * phi.squeeze() + PARA.rho * phi_t

        return l_E, p

    def bc_loss(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_x = self.calc_grad(phi, x) / PARA.cx
        l_B = torch.mean((-phi_x - torch.tensor(PARA.u0).to(device).to(
            torch.float32)) ** 2)
        return l_B

    def bc_rad_loss(self, model, x, t):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()
        phi_x = self.calc_grad(phi, x) / PARA.cx
        phi_t = self.calc_grad(phi, t) / PARA.ct
        u = -PARA.A * phi_x
        p = PARA.R * PARA.A * phi + PARA.rho * phi_t
        ut = self.calc_grad(u, t) / PARA.ct
        pt = self.calc_grad(p, t) / PARA.ct
        l_rad = torch.mean((PARA.rho * PARA.c / PARA.A * ut - model.rad_alpha * p - model.rad_beta * pt) ** 2)
        return l_rad

    def periodicity_loss(self, model, x0, t0, xT, tT):
        phi0 = model(x0, t0, PARA.alpha_phi)
        phi0 = phi0.squeeze()
        phiT = model(xT, tT, PARA.alpha_phi)
        phiT = phiT.squeeze()
        phi_x0 = self.calc_grad(phi0, x0) / PARA.cx
        phi_xT = self.calc_grad(phiT, xT) / PARA.cx
        phi_t0 = self.calc_grad(phi0, t0) / PARA.ct
        phi_tT = self.calc_grad(phiT, tT) / PARA.ct
        phi_tt0 = self.calc_grad(phi_t0, t0) / PARA.ct
        phi_ttT = self.calc_grad(phi_tT, tT) / PARA.ct

        l_Pu = torch.mean((-phi_x0 + phi_xT) ** 2)
        l_Pp = torch.mean(
            (PARA.R * PARA.A * phi0 + PARA.rho * phi_t0 - (PARA.R * PARA.A * phiT + PARA.rho * phi_tT)) ** 2)
        l_Pt = torch.mean((PARA.rho * phi_tt0 - PARA.rho * phi_ttT) ** 2)
        return l_Pu, l_Pp, l_Pt

    def obser_loss(self, model, x, t, p_end_infinite_flanged):
        phi = model(x, t, PARA.alpha_phi)
        phi = phi.squeeze()

        phi_t = self.calc_grad(phi, t) / PARA.ct
        p = PARA.R * PARA.A * phi + PARA.rho * phi_t

        l_O = torch.mean((p - p_end_infinite_flanged ) ** 2)

        return l_O

    def lr_schedule(self, lr, beta, epoch):
        lr = lr / (1 + beta * epoch)
        return lr

    def vis_p(self, xi_tot, ti_tot, p):
        plt.figure(figsize=(8, 6))

        # Convert tensors to numpy arrays
        xi_tot_np = xi_tot.detach().cpu().numpy()
        ti_tot_np = ti_tot.detach().cpu().numpy()
        p_np = p.detach().cpu().numpy()

        xi_tot_np = xi_tot_np / xi_tot_np.max()
        ti_tot_np = ti_tot_np / ti_tot_np.max()

        # Contour plot
        contour = plt.tricontourf(xi_tot_np, ti_tot_np, p_np, levels=1000, cmap='jet')
        plt.colorbar(contour, label='p values')
        plt.xlabel('x [m]', fontsize=12)
        plt.ylabel('t [s]', fontsize=12)
        plt.title('Predict pressure [Pa]', fontsize=14)

        plt.show()
    def save_model(self, model, main_optimizer,  epoch, tot_losses, E_losses, B_losses, P_losses,  O1_losses,  path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'main_optimizer_state_dict': main_optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'E_losses': E_losses,
            'B_losses': B_losses,
            'P_losses': P_losses,
            'O1_losses': O1_losses,
        }, path)

    def train(self):
        # get dataset
        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()


        # for epoch in range(n_epochs):
        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs", leave=True):
            main_optimizer.zero_grad()
            l_E, p = self.pde_loss(model, x_pde, t_pde)
            l_B = self.bc_loss(model, x_bc, t_bc)
            l_Pu, l_Pp, l_Pt = self.periodicity_loss(model, x_per0, t_per0, x_perT, t_perT)
            l_O1 = self.obser_loss(model, x_end_ob, t_end_ob, p_end_infinite_flanged)

            l_E = PARA.lambda_E * l_E
            l_B = PARA.lambda_B * l_B
            l_P = PARA.lambda_P * (PARA.lambda_u * l_Pu + PARA.lambda_p * l_Pp + PARA.lambda_t * l_Pt)
            l_O1 = 1 * PARA.lambda_P * PARA.lambda_p * l_O1

            loss = (l_E + l_B + l_P + l_O1 )

            loss.backward()
            main_optimizer.step()
            main_optimizer.param_groups[0]['lr'] = self.lr_schedule(PARA.lr, PARA.beta, epoch)

            self.tot_losses.append(loss.detach().cpu().item())
            self.E_losses.append(l_E.detach().cpu().item())
            self.B_losses.append(l_B.detach().cpu().item())
            self.P_losses.append(l_P.detach().cpu().item())
            self.O1_losses.append(l_O1.detach().cpu().item())

            if (epoch+1) % 100 == 0:
                print('Train\t Epoch: {:3} \tTotal Loss: {:.6f}'.format(epoch, loss))
                print('\t PDE Loss: {:.6f}'.format(l_E))
                print('\t BC Loss: {:.6f}'.format(l_B))
                print('\t Periodicity Loss: {:.6f}'.format(l_P))
                print('\t Observation Loss end: {:.6f}'.format(l_O1))

            del l_E, l_B, l_P, l_O1,  p
            torch.cuda.empty_cache()

            if (epoch+1) % 5000 == 0:

                save_path = (f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                             f'inverse_model_{epoch + 1}.pth')
                self.save_model(model, main_optimizer, epoch, self.tot_losses, self.E_losses, self.B_losses, self.P_losses,
                               self.O1_losses, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return model


    def load_trained_model(self, model, main_optimizer, path):
        # Load the checkpoint
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        main_optimizer.load_state_dict(checkpoint['main_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        tot_losses = checkpoint['tot_losses']
        E_losses = checkpoint['E_losses']
        B_losses = checkpoint['B_losses']
        P_losses = checkpoint['P_losses']
        O1_losses = checkpoint['O1_losses']

        return model, main_optimizer,epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses


    def plot_losses(self, tot_losses, E_losses, B_losses, P_losses):
        plt.figure(figsize=(12, 8))

        plt.plot(P_losses, label='P Loss', color='yellow', linestyle='-', linewidth=2)
        plt.plot( B_losses, label='B Loss', color='green', linestyle='-.', linewidth=2)
        plt.plot( E_losses, label='E Loss', color='lightblue', linestyle='-', linewidth=2)
        plt.plot( tot_losses, label='Total Loss', color='red', linewidth=2)
        plt.yscale('log')
        plt.title('Losses Over Epochs', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (log scale)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True)

        plt.tight_layout()
        plt.show()



    def vis_p_loc(self, t, p, loc):
        # TODO: Needed to be changed to your path
        obser_data_path = '/nas/home/xluan/thesis-xinmeng/PINN_wind/pinn_wind/main/data/observe_end_infinite_flanged_data.mat'
        data = loadmat(obser_data_path)
        p_end_infinite_flanged = data['p_end_infinite_flanged']

        plt.plot(t.detach().cpu().numpy(), p.detach().cpu().numpy(), label='PINN', color='green', linestyle='-.', linewidth=2)
        plt.plot(t.detach().cpu().numpy(), p_end_infinite_flanged.squeeze(), label='FD', color='blue',linestyle=':', linewidth=2)
        plt.title(f'p at the {loc}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def test(self, model_path):
        model, main_optimizer, epoch, tot_losses, E_losses, B_losses, P_losses,O1_losses = \
            self.load_trained_model(self.model, self.main_optimizer,  model_path)
        tot_losses = [loss for loss in tot_losses]
        E_losses = [loss for loss in E_losses]
        B_losses = [loss for loss in B_losses]
        P_losses = [loss for loss in P_losses]

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()

        u_t_end, p_end, p_t_end = self.get_end_for_radiation(model, x_coup, t_coup)

        data_to_save = {
            'p_end': p_end,
            'u_t_end': u_t_end,
            'p_t_end': p_t_end
        }
        # TODO: Needed to be changed to your path
        file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/radiation_for_predict_end_noise_snr_40.mat'
        scipy.io.savemat(file_name, data_to_save)

        print(f"Data saved to {file_name}")

    def save_model_lbfgs(self, model, optimizer, epoch, tot_losses, E_losses, B_losses, P_losses, R_losses, O1_losses, O2_losses, rad_alphas, rad_betas,  path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'E_losses': E_losses,
            'B_losses': B_losses,
            'P_losses': P_losses,
            'R_losses': R_losses,
            'O1_losses': O1_losses,
            # 'O2_losses': O2_losses,
            # 'p': p,
            'rad_alphas': rad_alphas,
            'rad_betas': rad_betas
        }, path)
    def continue_train(self, model_path):
        model, main_optimizer, epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses = \
            self.load_trained_model(self.model, self.main_optimizer, model_path)
        self.tot_losses = [loss for loss in tot_losses]
        self.E_losses = [loss for loss in E_losses]
        self.B_losses = [loss for loss in B_losses]
        self.P_losses = [loss for loss in P_losses]
        self.O1_losses = [loss for loss in O1_losses]

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

        def closure():
            optimizer.zero_grad()
            l_E, p = self.pde_loss(model, x_pde, t_pde)
            l_B = self.bc_loss(model, x_bc, t_bc)
            l_Pu, l_Pp, l_Pt = self.periodicity_loss(model, x_per0, t_per0, x_perT, t_perT)
            l_O1 = self.obser_loss(model, x_end_ob, t_end_ob, p_end_infinite_flanged)
            l_E = PARA.lambda_E * l_E
            l_B = PARA.lambda_B * l_B
            l_P = PARA.lambda_P * (PARA.lambda_u * l_Pu + PARA.lambda_p * l_Pp + PARA.lambda_t * l_Pt)
            l_O1 = 1 * PARA.lambda_P * PARA.lambda_p * l_O1

            loss = (l_E + l_B  + l_P + l_O1 )
            loss.backward()

            self.tot_losses.append(loss.detach().cpu().item())
            self.E_losses.append(l_E.detach().cpu().item())
            self.B_losses.append(l_B.detach().cpu().item())
            self.P_losses.append(l_P.detach().cpu().item())
            self.O1_losses.append(l_O1.detach().cpu().item())

            if (epoch+1) % 50 == 0:
                print('Train\t Epoch: {:3} \tTotal Loss: {:.6f}'.format(epoch, loss))
                print('\t PDE Loss: {:.6f}'.format(l_E))
                print('\t BC Loss: {:.6f}'.format(l_B))
                print('\t Periodicity Loss: {:.6f}'.format(l_P))
                print('\t Observation Loss end: {:.6f}'.format(l_O1))

            return loss


        for epoch in tqdm(range(3000), desc="Training Epochs", leave=True):
            loss = optimizer.step(closure)
            torch.cuda.empty_cache()

            if (epoch+1) % 100 == 0:
                # TODO: Needed to be changed to your path
                save_path = (
                    f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                    f'inverse_model_LBFGS_{epoch + 1}.pth')
                self.save_model(model, main_optimizer, epoch, self.tot_losses, self.E_losses, self.B_losses,
                                self.P_losses,
                                self.O1_losses, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return model


    def test_continue(self, model_path):
        model, main_optimizer, epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses = \
            self.load_trained_model(self.model, self.main_optimizer, model_path)

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()

        x_fd = torch.linspace(-1, 1, 5001).to(device)
        t_fd = torch.linspace(-1, 1, 1001).to(device)
        x_fd, t_fd = torch.meshgrid(x_fd, t_fd, indexing='ij')
        x_fd.requires_grad = True
        t_fd.requires_grad = True
        p_fd = []
        for index in range(x_fd.shape[1]):
            p_tmp = self.get_p(model, x_fd[:, index].squeeze(), t_fd[:, index].squeeze()).detach().cpu().numpy()
            p_fd.append(p_tmp)
        p_fd = np.array(p_fd).T

        data_to_save = {
                    'p_fd': p_fd
                }
        # TODO: Needed to be changed to your path
        file_name = '/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/p_all_predict_noise_snr_40_lbfgs.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(data_to_save, file)

        print(f"Data saved to {file_name}")
        print('')

    def load_trained_model_for_prediction(self,model, path):

        checkpoint = torch.load(path)

        # Restore the model and optimizer state dictionaries
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # Retrieve the epoch and loss values
        epoch = checkpoint['epoch']
        tot_losses = checkpoint['tot_losses']
        E_losses = checkpoint['E_losses']
        B_losses = checkpoint['B_losses']
        P_losses = checkpoint['P_losses']
        O1_losses = checkpoint['O1_losses']

        return model,  epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses

    def save_model_predict_rad(self, model, main_optimizer,  epoch, tot_losses, E_losses, B_losses, P_losses,  O1_losses, R_losses, rad_alphas, rad_betas, path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'main_optimizer_state_dict': main_optimizer.state_dict(),
            'epoch': epoch,
            'tot_losses': tot_losses,
            'E_losses': E_losses,
            'B_losses': B_losses,
            'P_losses': P_losses,
            'R_losses': R_losses,
            'O1_losses': O1_losses,
            'rad_alphas': rad_alphas,
            'rad_betas': rad_betas
        }, path)

    def continue_predict_radiation(self, model_path):
        model,epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses = \
            self.load_trained_model_for_prediction( self.model, model_path)
        self.tot_losses = [loss for loss in tot_losses]
        self.E_losses = [loss for loss in E_losses]
        self.B_losses = [loss for loss in B_losses]
        self.P_losses = [loss for loss in P_losses]
        self.O1_losses = [loss for loss in O1_losses]

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()
        addi_params = [model.rad_alpha, model.rad_beta]
        addi_params_ids = set(id(param) for param in addi_params)
        nn_params = [param for param in model.parameters() if id(param) not in addi_params_ids]

        lr_main = 1e-4 #TODO
        lr_addi = 1e-2
        main_optimizer = torch.optim.Adam(nn_params,lr=lr_main,weight_decay=0)
        addi_optimizer = torch.optim.Adam(addi_params, lr=lr_addi, weight_decay=0)

        for epoch in tqdm(range(self.n_epochs), desc="Training Epochs", leave=True):
            main_optimizer.zero_grad()
            addi_optimizer.zero_grad()

            l_E, p = self.pde_loss(model, x_pde, t_pde)
            l_B = self.bc_loss(model, x_bc, t_bc)
            l_R = self.bc_rad_loss(model, x_coup, t_coup)
            l_Pu, l_Pp, l_Pt = self.periodicity_loss(model, x_per0, t_per0, x_perT, t_perT)
            l_O1 = self.obser_loss(model, x_end_ob, t_end_ob, p_end_infinite_flanged)


            l_E = PARA.lambda_E * l_E
            l_B = PARA.lambda_B * l_B
            l_R = PARA.lambda_rad * l_R
            l_P = PARA.lambda_P * (PARA.lambda_u * l_Pu + PARA.lambda_p * l_Pp + PARA.lambda_t * l_Pt)
            l_O1 = 1 * PARA.lambda_P * PARA.lambda_p * l_O1


            loss = (l_E + l_B + l_P + l_O1 + l_R)

            loss.backward()
            main_optimizer.step()
            main_optimizer.param_groups[0]['lr'] = self.lr_schedule(lr_main, PARA.beta, epoch)

            addi_optimizer.step()
            addi_optimizer.param_groups[0]['lr'] = self.lr_schedule(lr_addi, PARA.beta, epoch)

            self.tot_losses.append(loss.detach().cpu().item())
            self.E_losses.append(l_E.detach().cpu().item())
            self.B_losses.append(l_B.detach().cpu().item())
            self.P_losses.append(l_P.detach().cpu().item())
            self.R_losses.append(l_R.detach().cpu().item())
            self.O1_losses.append(l_O1.detach().cpu().item())
            # self.O2_losses.append(l_O2.detach().cpu().item())
            self.rad_alphas.append(model.rad_alpha.detach().cpu().numpy())
            self.rad_betas.append(model.rad_beta.detach().cpu().numpy())

            if (epoch+1) % 50 == 0:
                print('Train\t Epoch: {:3} \tTotal Loss: {:.6f}'.format(epoch, loss))
                print('\t PDE Loss: {:.6f}'.format(l_E))
                print('\t BC Loss: {:.6f}'.format(l_B))
                print('\t Radiation Loss: {:.6f}'.format(l_R))
                print('\t Periodicity Loss: {:.6f}'.format(l_P))
                print('\t Observation Loss end: {:.6f}'.format(l_O1))
                print('\t alpha: {:.6f}'.format(model.rad_alpha.detach().cpu().numpy()))
                print('\t beta: {:.6f}'.format(model.rad_beta.detach().cpu().numpy()))

            del l_E, l_B, l_P, l_O1,  p
            torch.cuda.empty_cache()

            if (epoch+1) % 500 == 0:
                # TODO: Needed to be changed to your path
                save_path = (f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                             f'inverse_model_predict_radiation_2_{epoch + 1}.pth')
                self.save_model_predict_rad(model, main_optimizer, epoch, self.tot_losses,  self.E_losses, self.B_losses, self.P_losses,
                                            self.O1_losses, self.R_losses, self.rad_alphas, self.rad_betas, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return model

    def continue_predict_radiation_lbfgs(self, model_path):
        model, epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses = \
            self.load_trained_model_for_prediction(self.model, model_path)
        self.tot_losses = [loss for loss in tot_losses]
        self.E_losses = [loss for loss in E_losses]
        self.B_losses = [loss for loss in B_losses]
        self.P_losses = [loss for loss in P_losses]
        self.O1_losses = [loss for loss in O1_losses]

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

        def closure():
            optimizer.zero_grad()
            l_E, p = self.pde_loss(model, x_pde, t_pde)
            l_B = self.bc_loss(model, x_bc, t_bc)
            l_R = self.bc_rad_loss(model, x_coup, t_coup)
            l_Pu, l_Pp, l_Pt = self.periodicity_loss(model, x_per0, t_per0, x_perT, t_perT)
            l_O1 = self.obser_loss(model, x_end_ob, t_end_ob, p_end_infinite_flanged)

            l_E = PARA.lambda_E * l_E
            l_B = PARA.lambda_B * l_B
            l_R = PARA.lambda_rad * l_R
            l_P = PARA.lambda_P * (PARA.lambda_u * l_Pu + PARA.lambda_p * l_Pp + PARA.lambda_t * l_Pt)
            l_O1 = 1 * PARA.lambda_P * PARA.lambda_p * l_O1

            loss = (l_E + l_B  + l_P + l_O1 +l_R)

            loss.backward()

            self.tot_losses.append(loss.detach().cpu().item())
            self.E_losses.append(l_E.detach().cpu().item())
            self.B_losses.append(l_B.detach().cpu().item())
            self.P_losses.append(l_P.detach().cpu().item())
            self.R_losses.append(l_R.detach().cpu().item())
            self.O1_losses.append(l_O1.detach().cpu().item())
            self.rad_alphas.append(model.rad_alpha.detach().cpu().numpy())
            self.rad_betas.append(model.rad_beta.detach().cpu().numpy())

            if (epoch+1) % 10 == 0:
                print('Train\t Epoch: {:3} \tTotal Loss: {:.6f}'.format(epoch, loss))
                print('\t PDE Loss: {:.6f}'.format(l_E))
                print('\t BC Loss: {:.6f}'.format(l_B))
                print('\t Radiation Loss: {:.6f}'.format(l_R))
                print('\t Periodicity Loss: {:.6f}'.format(l_P))
                print('\t Observation Loss end: {:.6f}'.format(l_O1))
                print('\t alpha: {:.6f}'.format(model.rad_alpha.detach().cpu().numpy()))
                print('\t beta: {:.6f}'.format(model.rad_beta.detach().cpu().numpy()))


            return loss


        for epoch in tqdm(range(3000), desc="Training Epochs", leave=True):
            loss = optimizer.step(closure)
            torch.cuda.empty_cache()

            if (epoch+1) % 100 == 0:
                # TODO: Needed to be changed to your path
                save_path = (
                    f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                    f'inverse_model_predict_radiation_LBFGS_{epoch + 1}.pth')
                self.save_model_predict_rad(model, optimizer, epoch, self.tot_losses, self.E_losses, self.B_losses,
                                            self.P_losses,
                                            self.O1_losses, self.R_losses, self.rad_alphas, self.rad_betas, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return model

    def load_trained_model_for_prediction_2(self,model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        epoch = checkpoint['epoch']
        tot_losses = checkpoint['tot_losses']
        E_losses = checkpoint['E_losses']
        B_losses = checkpoint['B_losses']
        P_losses = checkpoint['P_losses']
        R_losses = checkpoint['R_losses']
        O1_losses = checkpoint['O1_losses']
        alphas = checkpoint['rad_alphas']
        betas = checkpoint['rad_betas']

        return model,  epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses, R_losses, alphas, betas

    def continue_predict_radiation_lbfgs_2(self, model_path):
        model, epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses, R_losses, alphas, betas = \
            self.load_trained_model_for_prediction_2(self.model, model_path)
        self.tot_losses = [loss for loss in tot_losses]
        self.E_losses = [loss for loss in E_losses]
        self.B_losses = [loss for loss in B_losses]
        self.P_losses = [loss for loss in P_losses]
        self.R_losses = [loss for loss in R_losses]
        self.O1_losses = [loss for loss in O1_losses]
        self.rad_alphas = [loss for loss in alphas]
        self.rad_betas = [loss for loss in betas]

        (x_pde, t_pde, x_bc, t_bc, x_coup, t_coup, x_per0, t_per0, x_perT, t_perT,
         x_end_ob, t_end_ob, p_end_infinite_flanged) = self.get_data()

        optimizer = torch.optim.LBFGS(model.parameters(), lr=1, max_iter=20, history_size=10)

        def closure():
            optimizer.zero_grad()
            l_E, p = self.pde_loss(model, x_pde, t_pde)
            l_B = self.bc_loss(model, x_bc, t_bc)
            l_R = self.bc_rad_loss(model, x_coup, t_coup)
            l_Pu, l_Pp, l_Pt = self.periodicity_loss(model, x_per0, t_per0, x_perT, t_perT)
            l_O1 = self.obser_loss(model, x_end_ob, t_end_ob, p_end_infinite_flanged)

            l_E = PARA.lambda_E * l_E
            l_B = PARA.lambda_B * l_B
            l_R = PARA.lambda_rad * l_R
            l_P = PARA.lambda_P * (PARA.lambda_u * l_Pu + PARA.lambda_p * l_Pp + PARA.lambda_t * l_Pt)
            l_O1 = 1 * PARA.lambda_P * PARA.lambda_p * l_O1

            loss = (l_E + l_B  + l_P + l_O1 +l_R)

            loss.backward()

            self.tot_losses.append(loss.detach().cpu().item())
            self.E_losses.append(l_E.detach().cpu().item())
            self.B_losses.append(l_B.detach().cpu().item())
            self.P_losses.append(l_P.detach().cpu().item())
            self.R_losses.append(l_R.detach().cpu().item())
            self.O1_losses.append(l_O1.detach().cpu().item())
            self.rad_alphas.append(model.rad_alpha.detach().cpu().numpy())
            self.rad_betas.append(model.rad_beta.detach().cpu().numpy())

            if (epoch+1) % 10 == 0:
                print('Train\t Epoch: {:3} \tTotal Loss: {:.6f}'.format(epoch, loss))
                print('\t PDE Loss: {:.6f}'.format(l_E))
                print('\t BC Loss: {:.6f}'.format(l_B))
                print('\t Radiation Loss: {:.6f}'.format(l_R))
                print('\t Periodicity Loss: {:.6f}'.format(l_P))
                print('\t Observation Loss end: {:.6f}'.format(l_O1))
                print('\t alpha: {:.6f}'.format(model.rad_alpha.detach().cpu().numpy()))
                print('\t beta: {:.6f}'.format(model.rad_beta.detach().cpu().numpy()))


            return loss


        for epoch in tqdm(range(3000), desc="Training Epochs", leave=True):
            loss = optimizer.step(closure)
            torch.cuda.empty_cache()

            if (epoch+1) % 100 == 0:
                # TODO: Needed to be changed to your path
                save_path = (
                    f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                    f'inverse_model_predict_radiation_continue_LBFGS_2_{epoch + 1}.pth')
                self.save_model_predict_rad(model, optimizer, epoch, self.tot_losses, self.E_losses, self.B_losses,
                                            self.P_losses,
                                            self.O1_losses, self.R_losses, self.rad_alphas, self.rad_betas, save_path)
                print(f'Model saved at epoch: {epoch + 1}.')

        return model

    def test_continue_predict_radiation_lbfgs(self, model_path):
        model, epoch, tot_losses, E_losses, B_losses, P_losses, O1_losses, R_losses, alphas, betas = \
            self.load_trained_model_for_prediction_2(self.model, model_path)
        self.tot_losses = [loss for loss in tot_losses]
        self.E_losses = [loss for loss in E_losses]
        self.B_losses = [loss for loss in B_losses]
        self.P_losses = [loss for loss in P_losses]
        self.R_losses = [loss for loss in R_losses]
        self.O1_losses = [loss for loss in O1_losses]
        self.rad_alphas = [loss for loss in alphas]
        self.rad_betas = [loss for loss in betas]

        epoch_index_1 = np.linspace(1,20000,20000)
        epoch_index_2 = np.linspace(20001, 21500, 49806-20000)
        epoch_index = np.concatenate([epoch_index_1, epoch_index_2])
        fig, axs = plt.subplots(2, 1, figsize=(12, 10))  # 2 rows, 1 column

        axs[0].plot(epoch_index, self.rad_alphas,  color='green', linestyle='-', linewidth=2)
        axs[0].set_xlabel('Epoch', fontsize=30)
        axs[0].set_ylabel(r'$\alpha$', fontsize=30)
        axs[0].grid(True, which='both')
        axs[0].tick_params(axis='x', labelsize=30)
        axs[0].tick_params(axis='y', labelsize=30)
        axs[0].axhline(y=1.2142, color='black', linestyle='-', linewidth=2)

        axs[1].plot(epoch_index, self.rad_betas,  color='green', linestyle='-', linewidth=2)
        axs[1].set_xlabel('Epoch', fontsize=30)
        axs[1].set_ylabel(r'$\beta$', fontsize=30)
        axs[1].grid(True, which='both')
        axs[1].tick_params(axis='x', labelsize=30)
        axs[1].tick_params(axis='y', labelsize=30)
        axs[1].axhline(y=0.7371, color='black', linestyle='-', linewidth=2)

        plt.tight_layout()
        # TODO: Needed to be changed to your path
        plt.savefig('/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/alpha_beta_epochs.png', dpi=300)
        plt.show()

def start( mode):
    if mode == 'predict_rad':
        print("Start predicting radiation...")
        import PINN_wind.pinn_wind.main.network.nn_fourier_openwind_rad_inverse as NN
        model = NN.MainNN(PARA.Nf, PARA.Nb).to(device)
        N_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")
        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0) #NOT USED
        trainer = Trainer(model, main_optimizer, 20002)

        epoch_test = 3000
        # TODO: Needed to be changed to your path
        trained_model_path = (f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                              f'inverse_model_LBFGS_{epoch_test}.pth')
        trainer.continue_predict_radiation(trained_model_path)

    elif mode == 'predict_rad_lbfgs':
        print("Start predicting radiation...")
        import PINN_wind.pinn_wind.main.network.nn_fourier_openwind_rad_inverse as NN
        model = NN.MainNN(PARA.Nf, PARA.Nb).to(device)
        N_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")
        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)  # NOT USED
        trainer = Trainer(model, main_optimizer, 20002)

        epoch_test = 3000
        # TODO: Needed to be changed to your path
        trained_model_path = (
            f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
            f'inverse_model_LBFGS_{epoch_test}.pth')
        trainer.continue_predict_radiation_lbfgs(trained_model_path)

    elif mode == 'predict_rad_lbfgs_2':
        print("Start predicting radiation...")
        import PINN_wind.pinn_wind.main.network.nn_fourier_openwind_rad_inverse as NN
        model = NN.MainNN(PARA.Nf, PARA.Nb).to(device)
        N_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")
        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)  # NOT USED
        trainer = Trainer(model, main_optimizer, 20002)

        epoch_test = 20000
        # TODO: Needed to be changed to your path
        trained_model_path  = (f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                     f'inverse_model_predict_radiation_2_{epoch_test}.pth')
        trainer.continue_predict_radiation_lbfgs_2(trained_model_path)

    elif mode == 'test_predict_rad_lbfgs':
        import PINN_wind.pinn_wind.main.network.nn_fourier_openwind_rad_inverse as NN
        model = NN.MainNN(PARA.Nf, PARA.Nb).to(device)
        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)  # NOT USED
        trainer = Trainer(model, main_optimizer, 20002)

        epoch_test = 1500
        # TODO: Needed to be changed to your path
        trained_model_path = (
            f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
            f'inverse_model_predict_radiation_continue_LBFGS_2_{epoch_test}.pth')
        trainer.test_continue_predict_radiation_lbfgs(trained_model_path)

    else:
        import PINN_wind.pinn_wind.main.network.nn_fourier_openwind_rad as NN
        print("Load data...")
        model = NN.MainNN(PARA.Nf, PARA.Nb).to(device)

        N_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {N_total_params}")
        N_nn_params = sum(param.numel() for param in model.parameters())
        print(f"NN trainable parameters: {N_nn_params}")

        main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
        trainer = Trainer(model, main_optimizer, 20002)

        if mode == 'train':
            print("Start training...")
            trained_model = trainer.train()
        elif mode == 'test':
            print("Start testing..=")
            # TODO: Needed to be changed to your path
            trained_model_path = (f'inverse_radiation/model/pinn_adam_20000epoch_stage1.pth')
            trainer.test(trained_model_path)

        elif mode == 'train_lbfgs':
            epoch_test = 20000
            # TODO: Needed to be changed to your path
            trained_model_path = (f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/'
                                         f'inverse_model_{epoch_test}.pth')
            trainer.continue_train(trained_model_path)
        elif mode == 'test_lbfgs':
            epoch_test = 1000
            # TODO: Needed to be changed to your path
            trained_model_path = (
                f'/nas/home/xluan/thesis-xinmeng/PINN_wind/result/fa2025/infinite_flanged/norad_noise_snr40/'
                f'inverse_model_LBFGS_{epoch_test}.pth')

            trainer.test_continue(trained_model_path)
    return

#'predict_rad', 'predict_rad_lbfgs', 'predict_rad_lbfgs_2', 'test_predict_rad_lbfgs','train', 'test','test_lbfgs'
mode = 'test'
print('Start.....')
start(mode)