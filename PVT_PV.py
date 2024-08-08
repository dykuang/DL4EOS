'''
Author @ 2024 Dongyang Kuang

Given PVTE data from statical thermal experiment and PVT data from Hugoniot experiment,
we want to predict the pressure and energy surface: P = P(V,T) and E = E(P,V) = E(V,T).
The two EOS surfaces are jointly learned by a neural network.

The unit with which the equation can hold?
    using standards units for P, V, E, T
    P:  1 GPa = 1e^9 Pa
    V:  1 A = 1e-10 m
    E:  1 eV = 1.6022e-19 J
    T:  1 K

NOTE: Before running the script:
    * Change TRAIN = True at line 33 to train your own model, remember to save model weights.
    * If Train = False, make sure to load the correct weights at line 293. Codes after line 293 are for evaluations and visualizations.
    * From line 189-286, be careful with the regularization terms (and their strength) in the training loop, make sure
      to change them accordingly with your actual regularizations depending on your task. 
    * condition for early stopping can be changed at line 284.
'''

#%%
from Networks import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import E_along_H, Get_T_from_PV
from sklearn.metrics import r2_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN = False
#%%
'''
Data preparation
'''
data = np.loadtxt('./data/data_PVTE.txt')

data_scale = np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True)
data_normalized = (data - np.min(data, axis=0, keepdims=True))/(np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
# E_SCALAR = 6.241509326*1e-3 # 1 eV = 6.241509326*1e-3 or 1e-2/1.6 eV 
# E_SCALAR = 1e-2/1.6022 # 1 eV = 1.6022e-19 J

# NOTE: though calculated, the E information is not used during the training
Vmin, Tmin, Pmin, Emin = np.min(data, axis=0).astype('float32')
Vmax, Tmax, Pmax, Emax = np.max(data, axis=0).astype('float32')
Vscale, Tscale, Pscale, Escale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin, Emax-Emin

# Hugoniont data
path = './data/VP_sharma.txt'
H_VP = pd.read_csv(path, sep='\s+', header=None)
# data = np.loadtxt(path)
# VP_H = H_VP[3:].to_numpy()
VP_H = H_VP[1:].to_numpy()
VP_H = VP_H.astype(np.float32)
# E_H = E_along_H(VP_H [:, 0], VP_H [:, 1], 
#                 V0=VP_H [0, 0], P0=VP_H [0, 1])

unique_VP_H, unique_indexes = np.unique(VP_H[:,0], return_index=True)

VP_H = VP_H[unique_indexes]
VP_H = VP_H[::4]

E_H = E_along_H(VP_H[:, 0], VP_H[:, 1], 
                V0=5.6648, P0=12, E0 = -8.9432)
# E_H = E_H*E_SCALAR # convert to eV

Vmin_both = 0.9*min(Vmin, np.min(VP_H[:,0]))
Vmax_both = max(Vmax, np.max(VP_H[:,0]))
Vscale_both = Vmax_both - Vmin_both

Pmin_both = min(Pmin, np.min(VP_H[:,1]))
Pmax_both = 1.1*max(Pmax, np.max(VP_H[:,1]))
Pscale_both = Pmax_both - Pmin_both

data_normalized[:,0] = (data[:,0] - Vmin_both)/Vscale_both
data_normalized[:,2] = (data[:,2] - Pmin_both)/Pscale_both

X = data_normalized[:,:2].astype(np.float32)
y = data_normalized[:,2:].astype(np.float32)

VP_H_normalized = np.c_[(VP_H[:,0] - Vmin_both)/Vscale_both, 
                        (VP_H[:,1] - Pmin_both)/Pscale_both]

plt.figure()
plt.plot( data[:,2], data[:,0], '.', label='PVT data')
plt.plot( VP_H[:,1], VP_H[:,0], '^', label='Hugoniot data')
plt.xlabel('Pressure (GPa)')
plt.ylabel(r'Volume ($\AA^{3}/atom$)')
# plt.title('P-V data')
plt.legend()


#%%
# class EOS_Dataset(Dataset):
#     def __init__(self, data):
#         self.X = torch.from_numpy(data[:,:2]).float()  # taking V and T
#         self.y = torch.from_numpy(data[:,2:]).float()  # taking P and V as target
        
#     def __len__(self):
#         return len(self.y)
    
#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]
    
# batch_size = 20
# train_dataset = EOS_Dataset(data_normalized)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, 
#                           shuffle=True)
#%%
'''
Define the Network Model
'''
# Define the input size, hidden size, and output size
input_size = 2
hidden_size = 64
output_size = 1

# Create an instance of the MLP network
# Pnet = Backbone(input_size, hidden_size, output_size)
# Enet = Backbone(input_size, hidden_size, output_size)

Pnet = Backbone_sep(hidden_size)
Enet = Backbone_sep(hidden_size)
# Enet = Backbone_sep_V1(hidden_size, feature_dim=4)

# Pnet = latent_basis_model(Xsize=X.shape[-1],
#                             latent_dim=2, Xlim=2, 
#                             hidden_size=8,num_blocks=4,
#                             l2_reg=1e-4,
#                             concateX=True)

# Enet = latent_basis_model(Xsize=X.shape[-1],
#                             latent_dim=4, Xlim=2, 
#                             hidden_size=16,num_blocks=4,
#                             l2_reg=1e-4,
#                             concateX=True)

Jnet = Joint_net(Pnet, Enet)

total_params = sum(p.numel() for p in Jnet.parameters())
print(f"Total number of parameters in Jnet: {total_params}")
#%%
'''
The training part
'''
import torch.optim as optim
# VT grid
Vspan = np.linspace(np.min(data_normalized[:,0]), 
                    np.max(data_normalized[:,0]), 20)
Tspan = np.linspace(np.min(data_normalized[:,1]), 
                    np.max(data_normalized[:,1]), 20)

VV, TT = np.meshgrid(Vspan, Tspan)
VT_grid = np.hstack((VV.reshape(-1,1), TT.reshape(-1,1)))
VT_grid = torch.from_numpy(VT_grid).float().to(device)
VT_grid.requires_grad = True # for computing the Jacobian

MSEloss = nn.MSELoss()
MAEloss = nn.L1Loss()

learning_rate = 5e-4
J_optimizer = optim.Adam(Jnet.parameters(), lr=learning_rate)
# P_optimizer = optim.Adam(Jnet.pnet.parameters(), lr=learning_rate)
# E_optimizer = optim.Adam(Jnet.enet.parameters(), lr=learning_rate)


epochs = 10000
XX = torch.tensor(X).to(device, dtype=torch.float32)
yy = torch.tensor(y).to(device, dtype=torch.float32)

# VP_hugo = torch.tensor(VP_H).to(device, dtype=torch.float32)
# E_hugo = torch.tensor(E_H[:,None]*E_SCALAR).to(device, dtype=torch.float32)
# NOTE: sharmar used meanH_unscaled = (meanE + 8.9431) + 0.5 * (X_testP[:,0] - 5.6648) * (meanP + 12)*((1e-2)/1.6)
VP_hugo = torch.tensor(VP_H_normalized).to(device, dtype=torch.float32)
E_hugo = torch.tensor(E_H[:,None]).to(device, dtype=torch.float32)

dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)
#%%
'''
training loop
'''
if TRAIN:
    for epoch in range(epochs):
        # for i, (X, y) in enumerate(train_loader):
        J_optimizer.zero_grad()
        P_pred = Jnet.pnet(XX)
        loss_P = MSEloss(P_pred, yy[:,:-1]) # use relative err?
        # loss_P = MSEloss(torch.ones_like(P_pred), P_pred/(1e-6+yy[:,:-1]))

        E_pred = Jnet.enet(VP_hugo)
        loss_E = MSEloss(E_pred, E_hugo)
        # loss_E = MSEloss(torch.ones_like(E_pred), E_pred/(1e-6+E_hugo))
        # Compute the Jacobian info on regular grid
        # NOTE: The Jacobian can also be computed using finite difference
        E_grid, P_grid = Jnet(VT_grid) # possible to use interpolation with grid data

        pE = torch.autograd.grad(E_grid, VT_grid, 
                                retain_graph = True,
                                create_graph = True,
                                grad_outputs = torch.ones_like(E_grid),
                                allow_unused = False)[0]
        
        pP = torch.autograd.grad(P_grid, VT_grid, 
                                retain_graph = True,
                                create_graph = True,
                                grad_outputs = torch.ones_like(P_grid),
                                allow_unused = False)[0]
        
        pEpV = pE[:,:1]  # should be smaller than 0
        pEpT = pE[:,1:]  # should be greater than 0
        pPpV = pP[:,:1]  # should be smaller than 0
        pPpT = pP[:,1:]  # should be greater than 0
    
        #----------------
        # Compute the Jacobian with finite difference
        #----------------
        # E_grid_pT, P_grid_pT = Jnet(VT_grid+delta_T)
        # E_grid_mT, P_grid_mT = Jnet(VT_grid-delta_T)
        # pPpT = (P_grid_pT - P_grid_mT)/(2.0*dT)
        # pEpT = (E_grid_pT - E_grid_mT)/(2.0*dT)

        # E_grid_pV, P_grid_pV = Jnet(VT_grid+delta_V)
        # E_grid_mV, P_grid_mV = Jnet(VT_grid-delta_V)    
        # pEpV = (E_grid_pV - E_grid_mV)/(2.0*dV)
        # pPpV = (P_grid_pV - P_grid_mV)/(2.0*dV)
        
        #-------------------------------------------------
        # Adding constraints
        #-------------------------------------------------
        # # Compute the loss on the Jacobian info
        # # NOTE: The loss's actual form depends on the specific unit of P, V, E and T
        # loss_D = MAEloss(P_grid*Pscale_both+Pmin_both, 
        #                 (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
        #                     - 1.6022e-2*pEpV/Vscale_both)
        # eqn_err = torch.abs(P_grid*Pscale_both+Pmin_both -  
        #                     ((VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
        #                     - 1.6022e-2*pEpV/Vscale_both)
        # )
        # loss_D = torch.max(eqn_err)
        
        '''
        # NOTE: adding a margin to help being away from 0
        '''
        loss_C = MSEloss(pEpT, torch.abs(pEpT)) \
                + MSEloss(-pPpV, torch.abs(pPpV)) \
                + MSEloss(pPpT, torch.abs(pPpT))  \
                + MSEloss( -pEpV, torch.abs(pEpV)) \
                

        #--------------------------------
        # Bulk Modulus loss
        #--------------------------------
        # BM = (VT_grid[:,:1]+Vmin_both/Vscale_both)*pPpV*Pscale_both
        # loss_BM = MSEloss(-(BM+200), torch.abs(BM+200))
        #--------------------------------
        # homogeneous loss-- zero order
        #--------------------------------
        # random_number = torch.randn(1).to(device)
        # E_grid_mul, P_grid_mul = Jnet(random_number*VT_grid)
        # loss_C = MSEloss(E_grid, E_grid_mul) + MSEloss(P_grid, P_grid_mul)
        
        loss = loss_P + loss_E + 1e-3*loss_C 

        # loss = loss_P + loss_E + 1e-3*loss_C + 0.0*loss_D + 1e-6*loss_BM
        # loss = loss_P + loss_E

        loss.backward()
        # loss_C.backward()
        J_optimizer.step()
        # if (epoch+1) % 10 == 0:
        #     print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}')

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}')
            # print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, \
            #         Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f},\
            #         Loss_D: {loss_D.item():.4f}')
        if loss_P.item() < 1e-4 and loss_E.item() < 1e-4:
        # if loss_P.item() < 1e-4 and loss_E.item() < 1e-4 and loss_C.item() < 1e-4:
            break

    # torch.save(Jnet.state_dict(), './weights/temp/exp_joint.pth') # uncomment to save the model

#%%
'''
NOTE: Load previously trained weights if needed, especially when TRAIN = False
'''
# Jnet.load_state_dict(torch.load('./weights/PVT_PV/exp_joint.pth'))

#%%
'''
Check the prediction
'''
E_pred, P_pred = Jnet(XX)
E_pred = E_pred.cpu().detach().numpy()
P_pred = P_pred.cpu().detach().numpy()

E_grid, P_grid = Jnet(VT_grid)
E_grid = E_grid.cpu().detach().numpy()
P_grid = P_grid.cpu().detach().numpy()

E_pred_H = Jnet.enet(VP_hugo)
E_pred_H = E_pred_H.cpu().detach().numpy()

from scipy.stats import pearsonr, spearmanr
# Calculate the R2 score for the predicted
Pr2 = r2_score(data[:, 2], P_pred.flatten()*Pscale_both+Pmin_both)
print(f"R2 score (P): {Pr2:.4f}")
P_corr, P_pval = pearsonr(data[:, 2], P_pred.flatten()*Pscale_both+Pmin_both)
print(f"Pearson (P): {P_corr:.4f}, p-val: {P_pval:.2e}")
P_corr_S, P_pval_S = spearmanr(data[:, 2], P_pred[:,0])
print(f"Spearman (P): {P_corr_S:.4f}, p-val: {P_pval_S:.2e}")

Er2H = r2_score(E_H, E_pred_H.flatten())
print(f"R2 score (E_H): {Er2H:.4f}")
E_corr_H, E_pval_H = pearsonr(E_H, E_pred_H.flatten())
print(f"Pearson (E_H): {E_corr_H:.4f}, p-val: {E_pval_H:.2e}")
E_corr_HS, E_pval_HS = spearmanr(data[:, -1], E_pred[:,0])
print(f"Spearman (E_H): {E_corr_HS:.4f}, p-val: {E_pval_HS:.2e}")

Er2 = r2_score(data[:, -1], E_pred.flatten())
print(f"R2 score (E): {Er2}")
E_corr_P, E_pval_P = pearsonr(data[:, -1], E_pred[:,0])
print(f"Pearson (E): {E_corr_P}, p-val: {E_pval_P:.2e}")
E_corr_S, E_pval_S = spearmanr(data[:, -1], E_pred[:,0])
print(f"Spearman (E): {E_corr_S:.4f}, p-val: {E_pval_S:.2e}")


#%%
from mpl_toolkits.mplot3d import Axes3D

fig, ax = plt.subplots(2,3)
# P 
ax[0,0].plot(data[:,-2], P_pred.flatten()*Pscale_both+Pmin_both, '+')
ax[0,0].plot(data[:,-2],data[:,-2], '--', color='r')
ax[0,0].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Pr2,P_corr,P_pval,P_corr_S,P_pval_S),
              transform=ax[0,0].transAxes, fontsize=14, 
              verticalalignment='top')
ax[0,0].set_xlabel('True Pressure (GPa)')
ax[0,0].set_ylabel('Predicted Pressure (GPa)')

ax[1,0].plot(data[:, -2], P_pred.flatten()*Pscale_both+Pmin_both - data[:,-2], '+', label='Error')
ax[1,0].axhline(y=0, color='r', linestyle='--')
idx = np.argsort(data[:, 2])
ax[1,0].fill_between(data[idx, 2],
                    -0.05*data[idx, 2],
                    0.05*data[idx, 2],
                    color='pink', alpha=0.4, label='5% error')
ax[1,0].set_xlabel('True Pressure (GPa)')
ax[1,0].set_ylabel('Error in Pressure (GPa)')
ax[1,0].legend()

# E on Hugoniont
ax[0,1].plot(E_H, E_pred_H.flatten(), '+')
ax[0,1].plot(E_H, E_H, '--', color='r')
ax[0,1].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Er2H,E_corr_H,E_pval_H,E_corr_HS,E_pval_HS),
             transform=ax[0,1].transAxes, fontsize=14, 
             verticalalignment='top')
ax[0,1].set_xlabel('True Pressure (GPa) - Hugoniot')
ax[0,1].set_ylabel('Predicted Energy (eV/atom) - Hugoniont')

# ax[1,1].plot(VP_H[:, 1], E_pred_H.flatten()-E_H, '+')
ax[1,1].plot(E_H, E_pred_H.flatten()-E_H, '+', label='Error')
idx = np.argsort(E_H)
ax[1,1].fill_between(E_H[idx],
                    -0.01*E_H[idx],
                    0.01*E_H[idx],
                    color='pink', alpha=0.4, label='1% error')
ax[1,1].axhline(y=0, color='r', linestyle='--')
ax[1,1].set_xlabel('True Pressure (GPa) - Hugoniot')
ax[1,1].set_ylabel('Error in Energy (eV/atom) - Hugoniot')
ax[1,1].legend()

# E scattered
ax[0,2].plot(data[:,-1], E_pred.flatten(), '+')
ax[0,2].plot(data[:,-1], data[:,-1], '--', color='r')
ax[0,2].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Er2,E_corr_P,E_pval_P,E_corr_S,E_pval_S),
             transform=ax[0,2].transAxes, fontsize=14, 
             verticalalignment='top')

ax[0,2].set_xlabel('True Energy (eV/atom)')
ax[0,2].set_ylabel('Predicted Energy (eV/atom)')

ax[1,2].plot(data[:, -1], E_pred.flatten() - data[:,-1], '+',label='Error')
idx = np.argsort(data[:, -1])
ax[1,2].fill_between(data[idx, -1],
                    -0.25*data[idx, -1],
                    0.25*data[idx, -1],
                    color='pink', alpha=0.4, label='30% error')
ax[1,2].axhline(y=0, color='r', linestyle='--')
ax[1,2].set_xlabel('True Energy (eV/atom)')
ax[1,2].set_ylabel('Error in Energy (eV/atom)')
ax[1,2].legend()
plt.tight_layout()

#%%
'''
a different plot
'''
# fig, ax = plt.subplots(1,3, figsize=(10,4))
# ax[0].plot( data[:,2], data[:,0], '.', label='PVT data')
# ax[0].plot( VP_H[:,1], VP_H[:,0], '^', label='Hugoniot data')
# ax[0].set_xlabel('Pressure (GPa)',size=12)
# ax[0].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
# ax[0].legend(fontsize=12)

# # P 
# ax[1].scatter(data[:,-2], P_pred.flatten()*Pscale_both+Pmin_both,
#               marker='o')
# ax[1].plot(data[:,-2], data[:,-2], '--', color='r')
# ax[1].set_xlabel('True Pressure (GPa)',size=12)
# ax[1].set_ylabel('Predicted Pressure (GPa)',size=12)

# idx = np.argsort(data[:, 2])
# ax[1].fill_between(data[idx, 2],
#                     0.95*data[idx, 2],
#                     1.05*data[idx, 2],
#                     color='pink', alpha=0.4, label='5% error')
# ax[1].legend(fontsize=12)

# # E scattered
# ax[2].scatter(data[:,-1], E_pred.flatten(), marker = 'o')
# ax[2].scatter(E_H, E_pred_H.flatten(), marker = '^' , s=100)
# ax[2].plot(data[:,-1], data[:,-1], '--', color='r')
# ax[2].set_xlabel('True Energy (eV/atom)',size=12)
# ax[2].set_ylabel('Predicted Energy (eV/atom)',size=12)

# idx = np.argsort(data[:, -1])
# ax[2].fill_between(data[idx, -1],
#                     0.75*data[idx, -1],
#                     1.25*data[idx, -1],
#                     color='pink', alpha=0.4, label='25% error')
# ax[2].set_xlabel('True Energy (eV/atom)',size=12)
# ax[2].set_ylabel('Error in Energy (eV/atom)',size=12)
# ax[2].legend(fontsize=12)
# plt.tight_layout()
#%%
'''
A different plot (2,2)
'''
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0][0].scatter( data[:,2], data[:,0], marker='o', label='PVT data')
ax[0][0].scatter( VP_H[:,1], VP_H[:,0], marker='^', c='k', s = 100,
                 alpha=0.5,
                 label='Hugoniot data')

ax[0][0].set_xlabel('Pressure (GPa)',size=12)
ax[0][0].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
ax[0][0].legend(fontsize=12)

ax[0][1].scatter(data[:,1], data[:,0],
                 label='PVT data')

ax[0][1].set_xlabel('Temperature (K)',size=12)
ax[0][1].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
ax[0][1].legend(fontsize=12)

# P 
ax[1][0].scatter(data[:,2], 
               P_pred.flatten()*Pscale_both+Pmin_both,
               marker='o')
ax[1][0].plot(data[:,-2], data[:,-2], '--', color='r', alpha=0.5)
ax[1][0].set_xlabel('True Pressure (GPa)',size=12)
ax[1][0].set_ylabel('Predicted Pressure (GPa)',size=12)

idx = np.argsort(data[:, 2])
ax[1][0].fill_between(data[idx, 2],
                    0.95*data[idx, 2],
                    1.05*data[idx, 2],
                    color='pink', alpha=0.4, label='5% error')
ax[1][0].legend(fontsize=12)

# E scattered
# ax[2].scatter(data[:,-1], E_pred.flatten(), marker = 'o')
ax[1][1].scatter(data[:,-1], 
               E_pred.flatten(),
               marker='o')
# ax[2].scatter(E_H, E_pred_H.flatten(), marker = '^' , 
#               s=100, c='orange')
ax[1][1].scatter(E_H, 
               E_pred_H.flatten(),
               marker='^', c= 'k', s=100)
ax[1][1].plot(data[:,-1], data[:,-1], '--', color='r', alpha=0.5)
ax[1][1].set_xlabel('True Energy (eV/atom)',size=12)
ax[1][1].set_ylabel('Predicted Energy (eV/atom)',size=12)

idx = np.argsort(data[:, -1])
ax[1][1].fill_between(data[idx, -1],
                    0.75*data[idx, -1],
                    1.25*data[idx, -1],
                    color='pink', alpha=0.4, 
                    label='25% error')
ax[1][1].set_xlabel('True Energy (eV/atom)',size=12)
ax[1][1].set_ylabel('Predicted Energy (eV/atom)',size=12)
ax[1][1].legend(fontsize=12)
plt.tight_layout()


#%%
# def P_predictor():
#     def func(x):
#         xx = torch.tensor(x).to(device, dtype=torch.float32)
#         p = Jnet.pnet(xx)
#         return p.cpu().detach().numpy()
#     return func
# Tspan = np.linspace(-2, 2, 100)
# T_H = []
# for i in range(len(VP_H_normalized)):
#     Temp = Get_T_from_PV(P_predictor(), 
#                          VP_H_normalized[i,0], Tspan, 
#                          VP_H_normalized[i,1])
#     # print(f'V: {VP_H[i,0]}, P: {VP_H[i,1]}, T: {Temp*Tscale+Tmin}')
#     T_H.append(Temp)
# T_H = np.array(T_H)

#%%
# %matplotlib qt
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection='3d')
# p1 = ax1.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin, 
#                  P_grid[:, 0]*Pscale_both+Pmin_both, 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
# Surface plot
ax1.plot_surface(VV*Vscale_both+Vmin_both, 
                TT*Tscale+Tmin, 
                P_grid.reshape((20,20))*Pscale_both+Pmin_both, 
                alpha=0.4, color='k',
                linewidth=0, label='Prediction'
                )
ax1.text2D(0.05, 0.95, "R2 score: {:.04f} \nCorrelation: {:.04f}".format(Pr2,P_corr), 
           transform=ax1.transAxes, fontsize=12, 
           verticalalignment='top')
ax1.scatter(data[:, 0], data[:, 1], data[:,2], linewidths=4,
            c='b', label='True')
# ax1.scatter(data[:, 0], data[:, 1], P_pred[:,0]*Pscale_both+Pmin_both,
#              linewidths=4, c='b', marker='^')
ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
ax1.set_zlabel('Pressure (GPa)', fontsize=15, labelpad=9)
# ax1.legend(['DFT-MD Data'], loc='upper right', fontsize=15)
# cbar1 = fig.colorbar(p0, ax=ax1, shrink=0.6)
# cbar1.set_label(r'$\sigma$', rotation=0, fontsize=15, labelpad=12)
# cbar1.ax.tick_params(labelsize=12)
# ax1.legend(loc='upper right', fontsize=15)

ax2 = fig.add_subplot(133, projection='3d')
# p0 = ax2.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin, 
#                  E_grid[:, 0], 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
ax2.plot_surface(VV*Vscale_both+Vmin_both, 
                TT*Tscale+Tmin, 
                E_grid.reshape((20,20)), 
                alpha=0.4, color='k',
                linewidth=0, label='Prediction'
                )
p1 = ax2.scatter(data[:, 0], data[:, 1], data[:,-1], 
                 c='r', linewidths=4, label='True')
ax2.text2D(0.05, 0.95, "R2 score: {:.04f} \nCorrelation: {:.04f}".format(Er2,E_corr_P), 
           transform=ax2.transAxes, fontsize=12, 
           verticalalignment='top')
ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
ax2.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)
# ax2.legend(loc='upper right', fontsize=15)
# cbar2 = fig.colorbar(p0, ax=ax2, shrink=0.6)
# cbar2.set_label(r'$\sigma$', rotation=0, fontsize=15, labelpad=12)
# cbar2.ax.tick_params(labelsize=12)

ax3 = fig.add_subplot(132, projection='3d')
# p0 = ax3.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  P_grid[:, 0]*Pscale_both+Pmin_both,
#                  E_grid[:, 0], 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
ax3.plot_surface(VV*Vscale_both+Vmin_both, 
                P_grid[:, 0].reshape((20,20))*Pscale_both+Pmin_both, 
                E_grid.reshape((20,20)), 
                alpha=0.4, color='k',
                linewidth=0, label='Prediction'
                )
ax3.text2D(0.05, 0.95, "R2 score: {:.04f} \nCorrelation: {:.04f}".format(Er2H,E_corr_H), 
           transform=ax3.transAxes, fontsize=12, 
           verticalalignment='top')
p1 = ax3.scatter(data[:, 0], data[:, 2], data[:,-1], 
                 c='r', linewidths=4, label='True')
p2 = ax3.scatter(VP_H[:,0], VP_H[:,1], E_H, 
                 c='b', linewidths=4, label='True along Hugoniot')
ax3.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax3.set_ylabel('Pressure (GPa)', fontsize=15, labelpad=12)
ax3.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)
# ax3.legend(loc='upper right', fontsize=15)

# Hugo = E_grid[:, 0] + \
#        E_SCALAR * 0.5 * (P_grid[:, 0]*Pscale_both+Pmin_both + VP_H[0, 1])\
#             *(- VP_H[0, 0] + VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both)

# Hugo = E_grid[:, 0] + 8.9431 +\
#        1e-2/1.6022 * 0.5 * (P_grid[:, 0]*Pscale_both+Pmin_both + 12)\
#             *(- 5.6648 + VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both)

# ax4 = fig.add_subplot(224, projection='3d')
# p0 = ax4.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin,
#                 #  P_grid[:, 0]*Pscale_both+Pmin_both,
#                  Hugo, 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")

# p1 = ax4.scatter(VP_H[:,0], 
#                 #  VP_H[:,1],
#                  T_H*Tscale+Tmin,
#                  np.zeros_like(VP_H[:,0]), 
#                  alpha=0.5, marker="^", color="r", 
#                  linewidths=4, label="Exp Data")
# ax4.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax4.set_ylabel('Temperature(K)', fontsize=15, labelpad=12)
# ax4.set_zlabel('Hugoniont (eV/atom)', fontsize=15, labelpad=8)

plt.tight_layout()

#%%
'''
forward pass 
'''
def forward_pass():
    # P_pred = Jnet.pnet(XX)
    # loss_P = MSEloss(P_pred, yy[:,:-1])

    # E_pred = Jnet.enet(VP_hugo)
    # loss_E = MSEloss(E_pred, E_hugo)
    
    # Compute the Jacobian info on regular grid
    # NOTE: The Jacobian can also be computed using finite difference
    E_grid, P_grid = Jnet(VT_grid) # possible to use interpolation with grid data

    pE = torch.autograd.grad(E_grid, VT_grid, 
                            retain_graph = True,
                            create_graph = True,
                            grad_outputs = torch.ones_like(E_grid),
                            allow_unused = False)[0]
    
    pP = torch.autograd.grad(P_grid, VT_grid, 
                            retain_graph = True,
                            create_graph = True,
                            grad_outputs = torch.ones_like(P_grid),
                            allow_unused = False)[0]
    
    pEpV = pE[:,:1]  # should be smaller than 0
    pEpT = pE[:,1:]  # should be greater than 0
    pPpV = pP[:,:1]  # should be smaller than 0
    pPpT = pP[:,1:]  # should be greater than 0
   
    eqn_err = P_grid*Pscale_both+Pmin_both -  \
                ( (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
                - 1.6022e-2*pEpV/Vscale_both )
    
    return eqn_err, pEpV, pEpT, pPpV, pPpT
    

#%%
'''
check the emprical density of partial derivatives
'''
eqn_err, pEpV, pEpT, pPpV, pPpT = forward_pass()
eqn_err = eqn_err.cpu().detach().numpy()
pEpV = pEpV.cpu().detach().numpy()
pEpT = pEpT.cpu().detach().numpy()
pPpV = pPpV.cpu().detach().numpy()
pPpT = pPpT.cpu().detach().numpy()

fig, ax = plt.subplots(2,2)
ax[0,0].hist(pEpV[:,0], bins=20, density=True)
ax[0,0].set_xlabel(r'$\frac{\partial E}{\partial V}$')
ax[0,1].hist(pEpT[:,0], bins=20, density=True)
ax[0,1].set_xlabel(r'$\frac{\partial E}{\partial T}$')
ax[1,0].hist(pPpV[:,0], bins=20, density=True)
ax[1,0].set_xlabel(r'$\frac{\partial P}{\partial V}$')
ax[1,1].hist(pPpT[:,0], bins=20, density=True)
ax[1,1].set_xlabel(r'$\frac{\partial P}{\partial T}$')
plt.tight_layout()


# %%
