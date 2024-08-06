'''
Author @ 2024 Dongyang Kuang

Evaluation on models trained from L2H_PVTE_train.py
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
# from sklearn.metrics import r2_score

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
# fig, ax = plt.subplots()
# ax.add_patch(Rectangle((0,0), 400, 4500,facecolor='b', alpha=0.4))
# ax.add_patch(Rectangle((0,0), 500, 5500,  facecolor='b', alpha=0.4))
# ax.add_patch(Rectangle((0,0), 650, 6500,  facecolor='r', alpha=0.4))
# ax.add_patch(Rectangle((0,0), 700, 7500,  facecolor='pink', alpha=0.4))
# ax.scatter(data[:,2], data[:,1])
# ax.text(20, 200, 'I', fontsize=14, family='serif', style='italic')
# ax.text(420, 200, 'II', fontsize=14, family='serif', style='italic')
# ax.text(520, 200, 'III', fontsize=14, family='serif', style='italic')
# ax.text(650, 200, 'IV', fontsize=14, family='serif', style='italic')
# plt.ylabel('Temperature (K)')
# plt.xlabel('Pressure (GPa)')

   
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
    
# torch.save(Jnet.state_dict(), 'CV_sharma_PVTE_init.pth')


#%%
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

# print("loading the trained weights...")
# Jnet.load_state_dict(torch.load('CV_PVTE_init.pth'))
# print("weights loaded.")

# %%
'''
make a summary for all the TYPE

only run the following after all the training for different TYPE
'''
summary = []
for type in ['I', 'II', 'III', 'IV']:
    Jnet.load_state_dict(torch.load('./weights/L2H/L2H_PVTE_{}.pth'.format(type)))
    if type == 'I':
        train_cond = np.logical_and(data[:,1]<=4500, data[:,2]<=400)
    elif type == 'II':
        train_cond = np.logical_and(data[:,1]<=5500, data[:,2]<=500)
    elif type == 'III':
        train_cond = np.logical_and(data[:,1]<=6500, data[:,2]<=650)
    elif type == 'IV':
        train_cond = np.logical_and(data[:,1]<=7500, data[:,2]<=700)
    else:
        assert 'TYPE must be I, II, III and IV'
    test_cond = ~train_cond
    with torch.no_grad():
        XX_test = torch.tensor(X[test_cond]).to(device, dtype=torch.float32)
        yy_test = torch.tensor(y[test_cond]).to(device, dtype=torch.float32)
        VP_test = torch.tensor(vp[test_cond]).to(device, dtype=torch.float32)

        P_pred_test = Jnet.pnet(XX_test)
        # E_pred_test = Jnet.enet(XX_test)
        # E_pred_test, P_pred_test = Jnet(XX_test)
        E_pred_test = Jnet.enet(VP_test)
        # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2

    loss_P_test = MSEloss(P_pred_test, yy_test[:,:-1])
    loss_P_test_mae = MAEloss(P_pred_test, yy_test[:,:-1])

    loss_E_test = MSEloss(E_pred_test, yy_test[:,-1:])
    loss_E_test_mae = MAEloss(E_pred_test, yy_test[:,-1:])
        
    summary.append((loss_P_test.item()**0.5*Pscale_both, #rmse
                    loss_E_test.item()**0.5*Escale_both,  # rmse
                    loss_P_test_mae.item()*Pscale_both, # mae
                    loss_E_test_mae.item()*Escale_both)) # mae))

summary = np.array(summary)
print('Summary for all the TYPE:')
print('RMSE-P, RMSE-E, MAE-P, MAE-E')
print(summary)

# %%
fig,ax = plt.subplots(1,3, figsize = (10,4))
ax[0].add_patch(Rectangle((0,0), 400, 4500,facecolor='b', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 500, 5500,  facecolor='b', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 650, 6500,  facecolor='r', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 700, 7500,  facecolor='pink', alpha=0.4))
ax[0].scatter(data[:,2], data[:,1])
ax[0].text(20, 200, 'I', fontsize=14, family='serif', style='italic')
ax[0].text(420, 200, 'II', fontsize=14, family='serif', style='italic')
ax[0].text(520, 200, 'III', fontsize=14, family='serif', style='italic')
ax[0].text(650, 200, 'IV', fontsize=14, family='serif', style='italic')
ax[0].set_ylabel('Temperature (K)')
ax[0].set_xlabel('Pressure (GPa)')

ax[1].plot(summary[:,0],'^--', label='Ours')
ax[1].set_xticks(np.arange(4))
ax[1].set_xticklabels(['I', 'II', 'III', 'IV'])
ax[1].set_ylabel('RMSE-P (GPa)')
ax[1].legend()

ax[2].plot(summary[:,1],'^--', label='Ours')
ax[2].set_xticks(np.arange(4))
ax[2].set_xticklabels(['I', 'II', 'III', 'IV'])
ax[2].set_ylabel('RMSE-E (eV/atom)')
ax[2].legend()
plt.tight_layout()

# %%
