'''
Author @ 2024 Dongyang Kuang

script for producing evaluations of trained models from CV_PVTE_res_train.py

Change the degree of the polynomial regression at line 108 or 
rewrite the this parameter as argument for the script using argparse.

'''
#%%
from Networks import *
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import pandas as pd
# from helper import E_along_H, Get_T_from_PV
# from sklearn.metrics import r2_score
import pickle
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
# E_SCALAR = 6.241509326*1e-3 # 1 eV = 6.241509326*1e-3 or 1e-2/1.6 eV 
# E_SCALAR = 1e-2/1.6022 # 1 eV = 1.6022e-19 J

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

# Tmin_both = Tmin
# Tmax_both = Tmax
# Tscale_both = Tmax_both - Tmin_both

# Emin_both = Emin
# Emax_both = Emax
# Escale_both = Emax_both - Emin_both

Vmin = Vmin - 2*Vscale/GRID_NUM
Vmax = Vmax + 2*Vscale/GRID_NUM
Vscale = Vmax - Vmin

Pmin = Pmin - 2*Pscale/GRID_NUM
Pmax = Pmax + 2*Pscale/GRID_NUM
Pscale = Pmax - Pmin

# Tmin = Tmin - 2*Tscale/GRID_NUM
# Tmax = Tmax + 2*Tscale/GRID_NUM
# Tscale = Tmax - Tmin

# Emin = Emin - 2*Escale/GRID_NUM
# Emax = Emax + 2*Escale/GRID_NUM
# Escale = Emax - Emin

data_normalized[:,0] = (data[:,0] - Vmin)/Vscale
data_normalized[:,2] = (data[:,2] - Pmin)/Pscale

# data_normalized[:,1] = (data[:,1] - Tmin)/Tscale
# data_normalized[:,-1] = (data[:,-1] - Emin)/Escale

X = data_normalized[:,:2].astype(np.float32)
y = data_normalized[:,2:].astype(np.float32)
vp = data_normalized[:,::2].astype(np.float32)

#%%
'''
polynomial features
'''
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
def poly_regression(x_train, y_train, degree, 
                    reg_model=LinearRegression()):
    
    poly = PolynomialFeatures(degree)
    
    poly_train = poly.fit_transform(x_train)

    reg_P = reg_model.fit(poly_train, y_train)   
    
    return reg_P, poly

def make_prediction(reg_model, poly, Xtest):
    test_features = poly.transform(Xtest)
    P_pred = reg_model.predict(test_features)
    return P_pred

degree = 1   # degree of base polynomial regression
#%%
# Define the input size, hidden size, and output size
input_size = 2
hidden_size = 32 # less hidden units accomodating less data and avoid overfitting
output_size = 1

# Create an instance of the MLP network
Pnet = Backbone(input_size, hidden_size, output_size)
Enet = Backbone(input_size, hidden_size, output_size)

# Pnet = Backbone_sep(hidden_size, droput_rate=0.0, 
#                     last_activation='linear')
# Enet = Backbone_sep(hidden_size, droput_rate=0.0, 
#                     last_activation='linear')
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
# VT grid
Vspan = np.linspace(np.min(data_normalized[:,0]), 
                    np.max(data_normalized[:,0]), GRID_NUM)
Tspan = np.linspace(np.min(data_normalized[:,1]), 
                    np.max(data_normalized[:,1]), GRID_NUM)

VV, TT = np.meshgrid(Vspan, Tspan)
VT_grid_np = np.hstack((VV.reshape(-1,1), TT.reshape(-1,1)))
VT_grid = torch.from_numpy(VT_grid_np).float().to(device)
VT_grid.requires_grad = True # for computing the Jacobian

MSEloss = nn.MSELoss()
MAEloss = nn.L1Loss()

dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)

eval_metricRE = []
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
for (train_index, test_index) in kf.split(X):
    Jnet.load_state_dict(torch.load('./weights/CV_PVTE_Poly/PVTE_fold{:02d}_res_d{}.pth'.format(fold,degree)))
    print('Fold: {}'.format(fold), 'loaded ...')
    Pmodel = LinearRegression()
    Preg_model, P_feature_trans = poly_regression(X[train_index], y[train_index,:1],
                                                  degree, Pmodel)
    P_pred_poly = torch.tensor( make_prediction(Preg_model, P_feature_trans, 
                                X[train_index])).to(device, dtype=torch.float32)
    P_pred_poly_test = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                       X[test_index])).to(device, dtype=torch.float32)
    P_pred_poly_grid = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                       VT_grid_np)).to(device, dtype=torch.float32)
    P_pred_poly_grid_pT = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                          VT_grid_np+np.array([[0, dT]]))).to(device, dtype=torch.float32)
    P_pred_poly_grid_mT = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                          VT_grid_np-np.array([[0, dT]]))).to(device, dtype=torch.float32)
    P_pred_poly_grid_pV = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                          VT_grid_np+np.array([[dV, 0]]))).to(device, dtype=torch.float32)
    P_pred_poly_grid_mV = torch.tensor(make_prediction(Preg_model, P_feature_trans, 
                                          VT_grid_np-np.array([[dV, 0]]))).to(device, dtype=torch.float32)
    

    
    Emodel = LinearRegression()
    Ereg_model, E_feature_trans = poly_regression(X[train_index], y[train_index,1:],
                                                  degree, Emodel)
    E_pred_poly = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                  X[train_index])).to(device, dtype=torch.float32)
    E_pred_poly_test = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                       X[test_index])).to(device, dtype=torch.float32)
    E_pred_poly_grid = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                       VT_grid_np)).to(device, dtype=torch.float32)
    E_pred_poly_grid_pT = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                          VT_grid_np+np.array([[0, dT]]))).to(device, dtype=torch.float32)
    E_pred_poly_grid_mT = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                          VT_grid_np-np.array([[0, dT]]))).to(device, dtype=torch.float32)
    E_pred_poly_grid_pV = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                          VT_grid_np+np.array([[dV, 0]]))).to(device, dtype=torch.float32)
    E_pred_poly_grid_mV = torch.tensor(make_prediction(Ereg_model, E_feature_trans, 
                                          VT_grid_np-np.array([[dV, 0]]))).to(device, dtype=torch.float32)
   

    with torch.no_grad():
        XX_test = torch.tensor(X[test_index]).to(device, dtype=torch.float32)
        yy_test = torch.tensor(y[test_index]).to(device, dtype=torch.float32)

        VP_test = torch.tensor(vp[test_index]).to(device, dtype=torch.float32)

        P_pred_test = Jnet.pnet(XX_test)
        # E_pred_test = Jnet.enet(XX_test)
        # E_pred_test, P_pred_test = Jnet(XX_test)
        E_pred_test = Jnet.enet(VP_test)
        # E_pred_test = 0.5*E_pred_test + 0.5*E_pred_test2

        loss_P_test = MSEloss(P_pred_test+ P_pred_poly_test, 
                              yy_test[:,:-1])
        loss_P_test_mae = MAEloss(P_pred_test+ P_pred_poly_test, 
                                  yy_test[:,:-1])
        loss_E_test = MSEloss(E_pred_test+ E_pred_poly_test, 
                              yy_test[:,-1:])
        loss_E_test_mae = MAEloss(E_pred_test+ E_pred_poly_test, 
                                  yy_test[:,-1:])
        
    eval_metricRE.append((loss_P_test.item()**0.5*Pscale, #rmse
                        loss_E_test.item()**0.5*Escale,  # rmse
                        loss_P_test_mae.item()*Pscale, # mae
                        loss_E_test_mae.item()*Escale, # mae)   
                        )
    )
    fold +=1
eval_metricRE=np.array(eval_metricRE)
print('RMSE_P, RMSE_E, MAE_P, MAE_E')
print( np.mean(eval_metricRE, axis=0) )

#%%
with open('CV_PVTE_record_res_d{}.pkl'.format(degree), 'wb') as f:
    pickle.dump(eval_metricRE, f)
