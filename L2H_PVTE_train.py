'''
Author @ 2024 Dongyang Kuang
Using Low T/P to predict High T/P for the EOS model

using 20 T, V, P, E data from Sharma, et all. 

NOTE: Before running the script
    * At line 73, change the TYPE to 'I', 'II', 'III', 'IV' to select the training data
    * Regularization strength can be adjusted at line 264
    * Change early stopping condition at line 278
'''

#%%
from Networks import *
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import pandas as pd
# from helper import E_along_H, Get_T_from_PV
from sklearn.metrics import r2_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
'''
Data preparation
'''
data = np.loadtxt('./data/data_PVTE.txt')
# data = pd.read_csv('diamond_PVTE.csv').to_numpy()  # if you want to use more data, comment the line above and uncomment this line
GRID_NUM = 20

data_scale = np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True)
data_normalized = (data - np.min(data, axis=0, keepdims=True))/(np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
# E_SCALAR = 6.241509326*1e-3 # 1 eV = 6.241509326*1e-3 or 1e-2/1.6 eV 
# E_SCALAR = 1e-2/1.6022 # 1 eV = 1.6022e-19 J

Vmin, Tmin, Pmin, Emin = np.min(data, axis=0).astype('float32')
Vmax, Tmax, Pmax, Emax = np.max(data, axis=0).astype('float32')
Vscale, Tscale, Pscale, Escale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin, Emax-Emin


Tmin_both = Tmin 
Tmax_both = Tmax 
Tscale_both = Tmax_both - Tmin_both

Emin_both = Emin 
Emax_both = Emax 
Escale_both = Emax_both - Emin_both

Vmin_both = Vmin - 2*Vscale/GRID_NUM # add some margin for the virtual grid
Vmax_both = Vmax + 2*Vscale/GRID_NUM
Vscale_both = Vmax_both - Vmin_both

Pmin_both = Pmin - 2*Pscale/GRID_NUM
Pmax_both = Pmax + 2*Pscale/GRID_NUM
Pscale_both = Pmax_both - Pmin_both

# Tmin_both = Tmin - 2*Tscale/GRID_NUM
# Tmax_both = Tmax + 2*Tscale/GRID_NUM
# Tscale_both = Tmax_both - Tmin_both

data_normalized[:,0] = (data[:,0] - Vmin_both)/Vscale_both
data_normalized[:,2] = (data[:,2] - Pmin_both)/Pscale_both

X = data_normalized[:,:2].astype(np.float32)
y = data_normalized[:,2:].astype(np.float32)
vp = data_normalized[:,::2].astype(np.float32)

#%%
'''
make the train/test partition based on (T, P) condition
'''
TYPE = 'IV' # 'I', 'II', 'III', 'IV', change this value to select training data
if TYPE == 'I':
    train_cond = np.logical_and(data[:,1]<=4500, data[:,2]<=400)
elif TYPE == 'II':
    train_cond = np.logical_and(data[:,1]<=5500, data[:,2]<=500)
elif TYPE == 'III':
    train_cond = np.logical_and(data[:,1]<=6500, data[:,2]<=650)
elif TYPE == 'IV':
    train_cond = np.logical_and(data[:,1]<=7500, data[:,2]<=700)
else:
    assert 'TYPE must be I, II, III and IV'
test_cond = ~train_cond

from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()
ax.add_patch(Rectangle((0,0), 400, 4500,facecolor='b', alpha=0.4))
ax.add_patch(Rectangle((0,0), 500, 5500,  facecolor='b', alpha=0.4))
ax.add_patch(Rectangle((0,0), 650, 6500,  facecolor='r', alpha=0.4))
ax.add_patch(Rectangle((0,0), 700, 7500,  facecolor='pink', alpha=0.4))
ax.scatter(data[:,2], data[:,1])
ax.text(20, 200, 'I', fontsize=14, family='serif', style='italic')
ax.text(420, 200, 'II', fontsize=14, family='serif', style='italic')
ax.text(520, 200, 'III', fontsize=14, family='serif', style='italic')
ax.text(650, 200, 'IV', fontsize=14, family='serif', style='italic')
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (GPa)')

   
#%%
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
    
# torch.save(Jnet.state_dict(), './temp/CV_sharma_PVTE_init.pth')


#%%
'''
The training part
'''
import torch.optim as optim
# VT grid
Vspan = np.linspace(np.min(data_normalized[:,0]), 
                    np.max(data_normalized[:,0]), GRID_NUM)
Tspan = np.linspace(np.min(data_normalized[:,1]), 
                    np.max(data_normalized[:,1]), GRID_NUM)

VV, TT = np.meshgrid(Vspan, Tspan)
VT_grid = np.hstack((VV.reshape(-1,1), TT.reshape(-1,1)))
VT_grid = torch.from_numpy(VT_grid).float().to(device)
VT_grid.requires_grad = True # for computing the Jacobian

MSEloss = nn.MSELoss()
MAEloss = nn.L1Loss()

learning_rate = 2e-4
J_optimizer = optim.Adam(Jnet.parameters(), lr=learning_rate)
# P_optimizer = optim.Adam(Jnet.pnet.parameters(), lr=learning_rate)
# E_optimizer = optim.Adam(Jnet.enet.parameters(), lr=learning_rate)

epochs = 20000
dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)

eval_metric = []

XX = torch.tensor(X[train_cond]).to(device, dtype=torch.float32)
yy = torch.tensor(y[train_cond]).to(device, dtype=torch.float32)

VP = torch.tensor(vp[train_cond]).to(device, dtype=torch.float32)
VP_test = torch.tensor(vp[test_cond]).to(device, dtype=torch.float32)

# print("loading the initial model...")
# Jnet.load_state_dict(torch.load('CV_sharma_PVTE_init.pth'))

#%%
'''
training loop
'''
history = []
for epoch in range(epochs):
    # for i, (X, y) in enumerate(train_loader):
    J_optimizer.zero_grad()
    # E_pred, P_pred = Jnet(XX)
    P_pred = Jnet.pnet(XX)
    E_pred = Jnet.enet(VP)
    # E_vp = Jnet.enet(VP)

    loss_P = MSEloss(P_pred, yy[:,:-1]) # use relative err?
    # loss_P = MSEloss(torch.ones_like(P_pred), P_pred/(1e-6+yy[:,:-1]))

    # E_pred = Jnet(XX)
    # loss_E = MSEloss(E_vp, yy[:,-1:])
    loss_E = MSEloss(E_pred, yy[:,-1:]) 
    # loss_E = 0.5*MSEloss(E_pred, yy[:,-1:]) + 0.5*MSEloss(E_vp, yy[:,-1:])
    # loss_E = MSEloss(torch.ones_like(E_pred), E_pred/(1e-6+E_hugo))
    # Compute the Jacobian info on regular grid
    # NOTE: The Jacobian can also be computed using finite difference
    E_grid, P_grid = Jnet(VT_grid) # possible to use interpolation with grid data

    # pE = torch.autograd.grad(E_grid, VT_grid, 
    #                         retain_graph = True,
    #                         create_graph = True,
    #                         grad_outputs = torch.ones_like(E_grid),
    #                         allow_unused = False)[0]
    
    # pP = torch.autograd.grad(P_grid, VT_grid, 
    #                         retain_graph = True,
    #                         create_graph = True,
    #                         grad_outputs = torch.ones_like(P_grid),
    #                         allow_unused = False)[0]
    
    # pEpV = pE[:,:1]  # should be smaller than 0
    # pEpT = pE[:,1:]  # should be greater than 0
    # pPpV = pP[:,:1]  # should be smaller than 0
    # pPpT = pP[:,1:]  # should be greater than 0

    #----------------
    # Compute the Jacobian with finite difference
    #----------------
    E_grid_pT, P_grid_pT = Jnet(VT_grid+delta_T)
    E_grid_mT, P_grid_mT = Jnet(VT_grid-delta_T)
    pPpT = (P_grid_pT - P_grid_mT)/(2.0*dT)
    pEpT = (E_grid_pT - E_grid_mT)/(2.0*dT)

    E_grid_pV, P_grid_pV = Jnet(VT_grid+delta_V)
    E_grid_mV, P_grid_mV = Jnet(VT_grid-delta_V)    
    pEpV = (E_grid_pV - E_grid_mV)/(2.0*dV)
    pPpV = (P_grid_pV - P_grid_mV)/(2.0*dV)
    
    #-------------------------------------------------
    # Adding constraints
    #-------------------------------------------------
    # # Compute the loss on the Jacobian info
    # # NOTE: The loss's actual form depends on the specific unit of P, V, E and T
    # loss_D = MAEloss(P_grid*Pscale_both+Pmin_both, 
    #                 (VT_grid[:,1:]+Tmin_both/Tscale_both)*pPpT*Pscale_both 
    #                     - 1.6022e-2*pEpV/Vscale_both*Escale_both)
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
            + MSEloss(-pEpV, torch.abs(pEpV)) \
    
    #--------------------------------
    # homogeneous loss-- zero order
    #--------------------------------
    # random_number = torch.randn(1).to(device)
    # E_grid_mul, P_grid_mul = Jnet(random_number*VT_grid)
    # loss_C = MSEloss(E_grid, E_grid_mul) + MSEloss(P_grid, P_grid_mul)

    loss = loss_P + loss_E + 1e-3*loss_C
    # loss = loss_P + loss_E

    loss.backward()
    # loss_C.backward()
    J_optimizer.step()
    # if (epoch+1) % 10 == 0:
    #     print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}')
    history.append(loss.item())
    if (epoch+1) % 100 == 0:
        # print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}')
        print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, \
                Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}')
    if loss_P.item() < 1e-4 and loss_E.item() < 1e-4 and loss_C.item() < 1e-3: 
        break

# torch.save(Jnet.state_dict(), './weights/temp/L2H_PVTE_{}.pth'.format(TYPE)) # uncomment this line to save the model
#%%
# Jnet.load_state_dict(torch.load('./weights/L2H/L2H_PVTE_{}.pth'.format(TYPE)))

with torch.no_grad():
    XX_test = torch.tensor(X[test_cond]).to(device, dtype=torch.float32)
    yy_test = torch.tensor(y[test_cond]).to(device, dtype=torch.float32)

    P_pred_test = Jnet.pnet(XX_test)
    # E_pred_test = Jnet.enet(XX_test)
    # E_pred_test, P_pred_test = Jnet(XX_test)
    E_pred_test = Jnet.enet(VP_test)
    # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2

loss_P_test = MSEloss(P_pred_test, yy_test[:,:-1])
loss_P_test_mae = MAEloss(P_pred_test, yy_test[:,:-1])

loss_E_test = MSEloss(E_pred_test, yy_test[:,-1:])
loss_E_test_mae = MAEloss(E_pred_test, yy_test[:,-1:])
    
eval_metric.append((loss_P_test.item()**0.5*Pscale_both, #rmse
                    loss_E_test.item()**0.5*Escale_both,  # rmse
                    loss_P_test_mae.item()*Pscale_both, # mae
                    loss_E_test_mae.item()*Escale_both, # mae
                    loss_C.item()))

print('RMSE_P, RMSE_E, MAE_P, MAE_E, Loss_C')
print(np.mean(np.array(eval_metric), axis=0))

# %%
with torch.no_grad():
    XX_test = torch.tensor(X[test_cond]).to(device, dtype=torch.float32)
    yy_test = torch.tensor(y[test_cond]).to(device, dtype=torch.float32)

    P_pred_test = Jnet.pnet(XX_test)
    # E_pred_test = Jnet.enet(XX_test)
    # E_pred_test, P_pred_test = Jnet(XX_test)
    E_pred_test = Jnet.enet(VP_test)
    # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2

loss_P_test = MSEloss(P_pred_test, yy_test[:,:-1])
loss_P_test_mae = MAEloss(P_pred_test, yy_test[:,:-1])

loss_E_test = MSEloss(E_pred_test, yy_test[:,-1:])
loss_E_test_mae = MAEloss(E_pred_test, yy_test[:,-1:])
    
eval_metric.append((loss_P_test.item()**0.5*Pscale_both, #rmse
                    loss_E_test.item()**0.5*Escale_both,  # rmse
                    loss_P_test_mae.item()*Pscale_both, # mae
                    loss_E_test_mae.item()*Escale_both # mae
                    ))

print('RMSE_P, RMSE_E, MAE_P, MAE_E')
print(np.mean(np.array(eval_metric), axis=0))
#%%
P_grid = P_grid.detach().cpu().numpy()
E_grid = E_grid.detach().cpu().numpy()

P_pred_test = P_pred_test.detach().cpu().numpy()
E_pred_test = E_pred_test.detach().cpu().numpy()

# %%
'''
gather some metrics on test
'''
from scipy.stats import pearsonr, spearmanr
# Calculate the R2 score for the predicted
Pr2 = r2_score(data[test_cond, 2], P_pred_test.flatten()*Pscale_both+Pmin_both)
print(f"R2 score (P): {Pr2:.4f}")
P_corr, P_pval = pearsonr(data[test_cond, 2], P_pred_test.flatten()*Pscale_both+Pmin_both)
print(f"Pearson (P): {P_corr:.4f}, p-val: {P_pval:.2e}")
P_corr_S, P_pval_S = spearmanr(data[test_cond, 2], P_pred_test[:,0])
print(f"Spearman (P): {P_corr_S:.4f}, p-val: {P_pval_S:.2e}")

Er2 = r2_score(data[test_cond, -1], E_pred_test.flatten()*Escale+Emin)
print(f"R2 score (E): {Er2}")
E_corr_P, E_pval_P = pearsonr(data[test_cond, -1], E_pred_test[:,0])
print(f"Pearson (E): {E_corr_P}, p-val: {E_pval_P:.2e}")
E_corr_S, E_pval_S = spearmanr(data[test_cond, -1], E_pred_test[:,0])
print(f"Spearman (E): {E_corr_S:.4f}, p-val: {E_pval_S:.2e}")
#%%
# %matplotlib qt
# fig = plt.figure(figsize=(18, 6))
# ax1 = fig.add_subplot(121, projection='3d')

# # Surface plot
# surface = ax1.plot_surface(VV*Vscale_both+Vmin_both, 
#                           TT*Tscale+Tmin, 
#                           P_grid.reshape((GRID_NUM,GRID_NUM))*Pscale_both+Pmin_both, 
#                           alpha=0.4, color='k',
#                           linewidth=0, label='Prediction'
#                           )
# p0 = ax1.scatter(data[train_cond, 0], 
#                  data[train_cond, 1], 
#                  data[train_cond, 2], linewidths=4,
#                  c='b', label='Train-True')

# ax1.scatter(data[test_cond, 0],
#             data[test_cond, 1],
#             data[test_cond, 2], linewidths=4,
#             c='r', label='Test-True')

# # ax1.scatter(data[test_cond, 0],
# #             data[test_cond, 1],
# #             P_pred_test[:,0]*Pscale_both+Pmin_both, 
# #             linewidths=4, marker='d', 
# #             c='r', label='Test-True')

# ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
# ax1.set_zlabel('Pressure (GPa)', fontsize=15, labelpad=9)
# ax1.legend(loc='upper right', fontsize=15)


# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot_surface(VV*Vscale_both+Vmin_both, 
#                  TT*Tscale+Tmin, 
#                  E_grid.reshape((GRID_NUM,GRID_NUM))*Escale+Emin, 
#                  alpha=0.4, color='k',
#                  linewidth=0, label='Prediction'
#                  )
# ax2.scatter(data[train_cond, 0], 
#             data[train_cond, 1], 
#             data[train_cond, -1], linewidths=4,
#             c='b', label='Train-True')

# ax2.scatter(data[test_cond, 0],
#             data[test_cond, 1],
#             data[test_cond, -1], linewidths=4,
#             c='r', label='Test-True')

# # ax2.scatter(data[test_cond, 0],
# #             data[test_cond, 1],
# #             E_pred_test[:,0]*Escale+Emin, 
# #             linewidths=4, marker='X',
# #             c='r', label='Test-True')

# ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
# ax2.set_zlabel('Energy (eV)', fontsize=15, labelpad=9)
# ax2.legend(loc='upper right', fontsize=15)