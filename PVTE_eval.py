'''
Author @ 2024 Dongyang Kuang

script for producing evaluations of trained models from CV_PVTE_train.py
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
Data preparation and normalization
'''
data = np.loadtxt('./data/data_PVTE.txt')
GRID_NUM = 20 # number of grids in the V-T space for virtual points

data_scale = np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True)
data_normalized = (data - np.min(data, axis=0, keepdims=True))/(np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))

Vmin, Tmin, Pmin, Emin = np.min(data, axis=0).astype('float32')
Vmax, Tmax, Pmax, Emax = np.max(data, axis=0).astype('float32')
Vscale, Tscale, Pscale, Escale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin, Emax-Emin

# Vmin, Tmin, Pmin, Emin = np.mean(data, axis=0).astype('float32')
# Vscale, Tscale, Pscale, Escale = np.std(data, axis=0).astype('float32')

# Vmin_both = Vmin
# Vmax_both = Vmax
# Vscale_both = Vmax_both - Vmin_both

# Pmin_both = Pmin
# Pmax_both = Pmax
# Pscale_both = Pmax_both - Pmin_both

Tmin_both = Tmin
Tmax_both = Pmax
Tscale_both = Tmax_both - Tmin_both

Emin_both = Emin
Emax_both = Emax
Escale_both = Emax_both - Emin_both

Vmin_both = Vmin - 2*Vscale/GRID_NUM
Vmax_both = Vmax + 2*Vscale/GRID_NUM
Vscale_both = Vmax_both - Vmin_both

Pmin_both = Pmin - 2*Pscale/GRID_NUM
Pmax_both = Pmax + 2*Pscale/GRID_NUM
Pscale_both = Pmax_both - Pmin_both

# Tmin_both = Tmin - 2*Tscale/GRID_NUM
# Tmax_both = Tmax + 2*Tscale/GRID_NUM
# Tscale_both = Tmax_both - Tmin_both

# Emin_both = Emin - 2*Escale/GRID_NUM
# Emax_both = Emax + 2*Escale/GRID_NUM
# Escale_both = Emax_both - Emin_both

data_normalized[:,0] = (data[:,0] - Vmin_both)/Vscale_both
data_normalized[:,2] = (data[:,2] - Pmin_both)/Pscale_both

# data_normalized[:,1] = (data[:,1] - Tmin_both)/Tscale_both
# data_normalized[:,-1] = (data[:,-1] - Emin_both)/Escale_both

X = data_normalized[:,:2].astype(np.float32)
y = data_normalized[:,2:].astype(np.float32)
vp = data_normalized[:,::2].astype(np.float32)


#%%
'''
Defining neural network models
'''
# Define the input size, hidden size, and output size
input_size = 2
hidden_size = 64
output_size = 1


Pnet = Backbone_sep(hidden_size, droput_rate=0.0, 
                    last_activation='linear')
Enet = Backbone_sep(hidden_size, droput_rate=0.0, 
                    last_activation='linear')


Jnet = Joint_net(Pnet, Enet)

total_params = sum(p.numel() for p in Jnet.parameters())
print(f"Total number of parameters in Jnet: {total_params}")

#%%
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

dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)

#%%
'''
confirm the evaluation and gather summery metrics from the trained models
'''
eval_metricRE = []
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42) # 5-fold CV, SEED must be the same as used in training
fold = 0
for (train_index, test_index) in kf.split(X):
    Jnet.load_state_dict(torch.load('./weights/CV_PVTE/CV_PVTE_fold{:02d}.pth'.format(fold)))
    print('Fold: {}'.format(fold), 'loaded ...')
    with torch.no_grad():
        XX_test = torch.tensor(X[test_index]).to(device, dtype=torch.float32)
        yy_test = torch.tensor(y[test_index]).to(device, dtype=torch.float32)

        VP_test = torch.tensor(vp[test_index]).to(device, dtype=torch.float32)

        # P_pred_test = Jnet.pnet(XX_test)
        # E_pred_test = Jnet.enet(XX_test)
        E_pred_test, P_pred_test = Jnet(XX_test)
        # E_pred_test = Jnet.enet(VP_test)
        # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2

        loss_P_test = MSEloss(P_pred_test, yy_test[:,:-1])
        loss_P_test_mae = MAEloss(P_pred_test, yy_test[:,:-1])
        
        loss_E_test = MSEloss(E_pred_test, yy_test[:,-1:])
        loss_E_test_mae = MAEloss(E_pred_test, yy_test[:,-1:])
        
    eval_metricRE.append((loss_P_test.item()**0.5*Pscale_both, #rmse
                        loss_E_test.item()**0.5*Escale_both,  # rmse
                        loss_P_test_mae.item()*Pscale_both, # mae
                        loss_E_test_mae.item()*Escale_both, # mae)   
                        )
    )
    fold +=1
eval_metricRE=np.array(eval_metricRE)
print('RMSE_P, RMSE_E, MAE_P, MAE_E')
print( np.mean(eval_metricRE, axis=0) )
print( np.std(eval_metricRE, axis=0) )
# %%
'''
show plots on fold1
'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for (train_index, test_index) in kf.split(X):
    Jnet.load_state_dict(torch.load('./weights/CV_PVTE/CV_PVTE_fold{:02d}.pth'.format(fold)))
    print('Fold: {}'.format(fold), 'loaded ...')
    fold += 1
    with torch.no_grad():

        XXX = torch.tensor(X).to(device, dtype=torch.float32)
        yyy = torch.tensor(y).to(device, dtype=torch.float32)
        # VPall = torch.tensor(vp).to(device, dtype=torch.float32)
        # P_pred_test = Jnet.pnet(XX_test)
        # E_pred_test = Jnet.enet(XX_test)
        E_pred_all, P_pred_all = Jnet(XXX)
        # E_pred_test = Jnet.enet(VP_test)
        # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2
    if fold>0:
        break

E_pred_all = E_pred_all.detach().numpy()[:,0]
P_pred_all = P_pred_all.detach().numpy()[:,0]

E_err = (E_pred_all - y[:, 1])*Escale_both
P_err = (P_pred_all - y[:, 0])*Pscale_both
# %%
idx = np.argsort(y[:,0])
train_idx_sorted = [ii for ii in idx if ii in train_index]
test_idx_sorted = [ii for ii in idx if ii in test_index]

P_sorted = np.sort(data[:,2])
E_sorted = np.sort(data[:,3])
fig, ax = plt.subplots(1,2)
ax[0].scatter(data[train_index,2], 
              P_pred_all[train_index]*Pscale_both+Pmin_both,
              marker='o', s=24, c='b', label='Train')
ax[0].scatter(data[test_index,2], 
              P_pred_all[test_index]*Pscale_both+Pmin_both,
              marker='x', s=34, c='r', label='Test')
ax[0].plot(P_sorted, P_sorted, 'k--')
ax[0].fill_between(P_sorted, 0.95*P_sorted,
                   1.05*P_sorted, color='pink', alpha=0.5,
                   label='%5 error')
ax[0].set_xlabel('True Pressure (Gpa)')
ax[0].set_ylabel('Predicted Pressure (Gpa)')
ax[0].legend()
ax[1].scatter(data[train_index,3], 
              E_pred_all[train_index]*Escale_both+Emin_both,
              marker='o', s=24, c='b', label='Train')
ax[1].scatter(data[test_index,3], 
              E_pred_all[test_index]*Escale_both+Emin_both,
              marker='x', s=34, c='r', label='Test')
ax[1].plot(E_sorted, E_sorted, 'k--')
ax[1].fill_between(E_sorted, 0.95*E_sorted,
                   1.05*E_sorted, color='pink', alpha=0.5,
                   label='%5 Error')
ax[1].set_ylabel('Predicted Energy (eV/atom)')
ax[1].set_xlabel('True Energy (eV/atom)')
ax[1].legend()
plt.tight_layout()

# %%
