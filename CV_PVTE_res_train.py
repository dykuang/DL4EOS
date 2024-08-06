'''
Author @ 2024 Dongyang Kuang
Cross Validation for the EOS model using PVTE data -- i.e. P and E predictions are jointly supervised.

Similar to CV_PVTE_train.py, but with polynomial regression as the base regressor and
Neural Network for the residual. 

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
hidden_size = 32 # less hidden units accomodating the scenario that we already have the first stage regression
output_size = 1

# Create an instance of the MLP network, since we are fitting the residual and not enforcing particular 
# structure as Backbone_sep. 
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
    
torch.save(Jnet.state_dict(), 'CV_PVTE_init_res.pth')


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
VT_grid_np = np.hstack((VV.reshape(-1,1), TT.reshape(-1,1)))
VT_grid = torch.from_numpy(VT_grid_np).float().to(device)
VT_grid.requires_grad = True # for computing the Jacobian

MSEloss = nn.MSELoss()
MAEloss = nn.L1Loss()

learning_rate = 5e-4
J_optimizer = optim.Adam(Jnet.parameters(), lr=learning_rate)
# P_optimizer = optim.Adam(Jnet.pnet.parameters(), lr=learning_rate)
# E_optimizer = optim.Adam(Jnet.enet.parameters(), lr=learning_rate)

epochs = 20000
dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)

#%%
'''
CV - split
'''
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 0
CV_loss = []

CV_ind_train = []
CV_ind_test = []

eval_metric = []

for (train_index, test_index) in kf.split(X):
    print("Fold: {} started ...".format(fold))
    XX = torch.tensor(X[train_index]).to(device, dtype=torch.float32)
    yy = torch.tensor(y[train_index]).to(device, dtype=torch.float32)

    VP = torch.tensor(vp[train_index]).to(device, dtype=torch.float32)

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
   
    CV_ind_train.append(train_index)
    CV_ind_test.append(test_index)
    
    print("loading the initial model...")
    Jnet.load_state_dict(torch.load('CV_PVTE_init_res.pth'))

    '''
    training loop
    '''
    ind_shuffle = np.arange(len(train_index))
    history = []
    for epoch in range(epochs):
        # if epoch % 100 == 0:
        #     np.random.shuffle(ind_shuffle )
        J_optimizer.zero_grad()
        # E_pred, P_pred = Jnet(XX[ind_shuffle])
        P_pred = Jnet.pnet(XX[ind_shuffle])
        E_pred = Jnet.enet(VP[ind_shuffle])

        loss_P = MSEloss(P_pred+P_pred_poly[ind_shuffle], 
                         yy[ind_shuffle,:-1]) # use relative err?
        # loss_P = MSEloss(torch.ones_like(P_pred), P_pred/(1e-6+yy[:,:-1]))

        # E_pred = Jnet(XX)
        # loss_E = MSEloss(E_vp, yy[:,-1:])
        loss_E = MSEloss(E_pred+E_pred_poly[ind_shuffle], 
                         yy[ind_shuffle,-1:]) 
        # loss_E = 0.5*MSEloss(E_pred, yy[:,-1:]) + 0.5*MSEloss(E_vp, yy[:,-1:])
        # loss_E = MSEloss(torch.ones_like(E_pred), E_pred/(1e-6+E_hugo))
        # Compute the Jacobian info on regular grid
        # NOTE: The Jacobian can also be computed using finite difference
        E_grid, P_grid = Jnet(VT_grid) # possible to use interpolation with grid data

   
        #----------------
        # Compute the Jacobian with finite difference
        #----------------
        E_grid_pT, P_grid_pT = Jnet(VT_grid+delta_T)
        E_grid_mT, P_grid_mT = Jnet(VT_grid-delta_T)
        pPpT = (P_grid_pT + P_pred_poly_grid_pT- P_grid_mT - P_pred_poly_grid_mT)/(2.0*dT)
        pEpT = (E_grid_pT + E_pred_poly_grid_pT- E_grid_mT - E_pred_poly_grid_mT)/(2.0*dT)

        E_grid_pV, P_grid_pV = Jnet(VT_grid+delta_V)
        E_grid_mV, P_grid_mV = Jnet(VT_grid-delta_V)    
        pEpV = (E_grid_pV + E_pred_poly_grid_pV- E_grid_mV-E_pred_poly_grid_mV)/(2.0*dV)
        pPpV = (P_grid_pV + P_pred_poly_grid_pV- P_grid_mV-P_pred_poly_grid_mV)/(2.0*dV)
        
        #-------------------------------------------------
        # Adding constraints
        #-------------------------------------------------
        # # Compute the loss on the Jacobian info
        # # NOTE: The loss's actual form depends on the specific unit of P, V, E and T
        # loss_D = MAEloss(P_grid*Pscale+Pmin, 
        #                 (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale 
        #                     - 1.6022e-2*pEpV/Vscale*Escale)
        # eqn_err = torch.abs(P_grid*Pscale_both+Pmin_both -  
        #                     ((VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
        #                     - 1.6022e-2*pEpV/Vscale_both)
        # )
        # loss_D = torch.max(eqn_err)
        
        '''
        # NOTE: adding a margin to help being away from 0
        '''
        loss_C = MSEloss( -pEpV, torch.abs(pEpV)) \
                + MSEloss(pPpT, torch.abs(pPpT))  \
                + MSEloss(pEpT, torch.abs(pEpT)) \
                + MSEloss(-pPpV, torch.abs(pPpV))
        
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
            print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}')
            # print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, \
            #         Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f},\
            #         Loss_D: {loss_D.item():.4f}')
        if loss_P.item() < 5e-6 and loss_E.item() < 5e-6:
            break
        # if fold > 0:
        #     break
    
    print('Training finished at fold: {}'.format(fold))
    with torch.no_grad():
        XX_test = torch.tensor(X[test_index]).to(device, dtype=torch.float32)
        yy_test = torch.tensor(y[test_index]).to(device, dtype=torch.float32)

        # P_pred_test = Jnet.pnet(XX_test)
        # E_pred_test = Jnet.enet(XX_test)
        E_pred_test, P_pred_test = Jnet(XX_test)

        loss_P_test = MSEloss(P_pred_test + P_pred_poly_test, 
                              yy_test[:,:-1])
        loss_P_test_mae = MAEloss(P_pred_test+ P_pred_poly_test,
                                  yy_test[:,:-1])
        
        loss_E_test = MSEloss(E_pred_test + E_pred_poly_test, 
                              yy_test[:,-1:])
        loss_E_test_mae = MAEloss(E_pred_test+ E_pred_poly_test, 
                                  yy_test[:,-1:])
        
    eval_metric.append((loss_P_test.item()**0.5*Pscale, #rmse
                        loss_E_test.item()**0.5*Escale,  # rmse
                        loss_P_test_mae.item()*Pscale, # mae
                        loss_E_test_mae.item()*Escale, # mae
                        loss_C.item()))
    

    torch.save(Jnet.state_dict(), 'PVTE_fold{:02d}_res_d{}.pth'.format(fold,degree))
    fold += 1
    CV_loss.append(history)

#%%

import pickle
CV_record = {'CV_loss': CV_loss, 
             'CV_ind_train': CV_ind_train, 
             'CV_ind_test': CV_ind_test, 
             'eval_metric': np.array(eval_metric)}
print(np.mean(CV_record['eval_metric'], axis=0))
print(np.std(CV_record['eval_metric'], axis=0))
#%%
# with open('CV_sharma_PVTE_record_res{}_{}.pkl'.format(DATE,degree), 'wb') as f:
#     pickle.dump(CV_record, f)

#%%
# with open('CV_sharma_record.pkl', 'rb') as f:
#     CV_record = pickle.load(f)

# CV_loss = CV_record['CV_loss']
# eval_metric = CV_record['eval_metric']
# CV_ind_test = CV_record['CV_ind_test']
# CV_ind_train = CV_record['CV_ind_train']

# %%
