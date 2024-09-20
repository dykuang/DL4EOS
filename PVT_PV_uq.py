'''
Author @ 2024 Dongyang Kuang

Adding uncertainty to the deterministic EOS model from `PVT_PV.py`

* aleatoric uncertainty is considered via outputing a distribution
* epistemic uncertainty is considered via dropout/variational layer.

NOTE: Before running the script:
    * Change TRAIN = True at line 40 to train your own model, remember to save model weights.
    * If Train = False, make sure to load the correct weights at line 442. Codes after line 442 are for evaluations and visualizations.
    * Key parameters that affects ways of incorporating uncertainties (as well as the ouput) are from line 122 to line 127.
      Be sure to check relevant loss, regularization and hyperparameters at the training block (line 283 - 432) accordingly based on your choice.
    * Artifical noise added to data (both input and output) can be controlled at line 198. Together with the above key parameters, they can
      affect the model's performance and the uncertainty estimation. 
    * Because of the nature randomness in dropout, the results may vary between runs if dropout is involved. The major patterns should be consistent.
      We saved the results from multiple runs in the summary folder (uq_saved_sample.pkl) if one would like to reproduce relevant plots in the paper. 
'''
#%%
from Networks import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from helper import E_along_H, Get_T_from_PV
from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

const_bnn_prior_parameters = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Reparameterization",  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
} # configurations for converting a deterministic model to a Bayesian model

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN = False # NOTE: Change to True to train your own model
#%%
'''
Data preparation
'''
data = np.loadtxt('./data/data_PVTE.txt')

data_scale = np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True)
data_normalized = (data - np.min(data, axis=0, keepdims=True))/(np.max(data, axis=0, keepdims=True) - np.min(data, axis=0, keepdims=True))
# E_SCALAR = 6.241509326*1e-3 # 1 eV = 6.241509326*1e-3 or 1e-2/1.6 eV 
# E_SCALAR = 1e-2/1.6022 # 1 eV = 1.6022e-19 J

Vmin, Tmin, Pmin, Emin = np.min(data, axis=0).astype('float32')
Vmax, Tmax, Pmax, Emax = np.max(data, axis=0).astype('float32')

Vscale, Tscale, Pscale, Escale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin, Emax-Emin

# Hugoniot data
path = './data/VP_sharma.txt'
H_VP = pd.read_csv(path, sep='\s+', header=None)
# data = np.loadtxt(path)
# VP_H = H_VP[3:].to_numpy()
VP_H = H_VP[1:].to_numpy()
VP_H = VP_H.astype(np.float32)

unique_VP_H, unique_indexes = np.unique(VP_H[:,0], return_index=True)

VP_H = VP_H[unique_indexes]
VP_H = VP_H[::4]

# E_H = E_along_H(VP_H [:, 0], VP_H [:, 1], 
#                 V0=VP_H [0, 0], P0=VP_H [0, 1])
E_H = E_along_H(VP_H[:, 0], VP_H[:, 1], 
                V0=5.6648, P0=12, E0 = -8.9432)


Vmin_both = 0.9*min(Vmin, np.min(VP_H[:,0]))
Vmax_both = max(Vmax, np.max(VP_H[:,0]))
Vscale_both = Vmax_both - Vmin_both

Pmin_both = min(Pmin, np.min(VP_H[:,1]))
Pmax_both = 1.1*max(Pmax, np.max(VP_H[:,1]))
Pscale_both = Pmax_both - Pmin_both

data_normalized[:,0] = (data[:,0] - Vmin_both)/Vscale_both
data_normalized[:,2] = (data[:,2] - Pmin_both)/Pscale_both

X = data_normalized[:,:2].astype(np.float32)
y = data_normalized[:,2:3].astype(np.float32)

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
    
# batch_size = 200
# train_dataset = EOS_Dataset(data_normalized)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, 
#                           shuffle=True)
#%%
# Define the input size, hidden size, and output size
input_size = 2
hidden_size = 64
output_size = 1
dropout_rate = 0.01 # NOTE: dropout rate for dropout model, can affect the uncertainty estimation

OUT_MODE = 'prob' # 'regular' or 'prob'
# OUT_MODE = 'regular'
# EP_MODE = 'variational' #
EP_MODE = 'dropout' # 'variational' or 'dropout'

# Create an instance of the MLP network
if OUT_MODE == 'prob' and EP_MODE == 'variational':
    Pnet = Backbone_sep_prob(hidden_size)
    Enet = Backbone_sep_prob(hidden_size)
    Jnet = Joint_net_prob(Pnet, Enet)
    dnn_to_bnn(Jnet, const_bnn_prior_parameters) # convert to Bayesian model

elif OUT_MODE == 'prob' and EP_MODE == 'dropout':
    '''
    Need a dropout layer in prob model
    '''
    Pnet = Backbone_sep_prob(hidden_size, dropout_rate = dropout_rate)
    Enet = Backbone_sep_prob(hidden_size, dropout_rate = dropout_rate)
    Jnet = Joint_net_prob(Pnet, Enet)

elif OUT_MODE == 'regular' and EP_MODE == 'variational':
    Pnet = Backbone_sep(hidden_size)
    Enet = Backbone_sep(hidden_size)
    Jnet = Joint_net(Pnet, Enet)
    dnn_to_bnn(Jnet, const_bnn_prior_parameters)

elif OUT_MODE == 'regular' and EP_MODE == 'dropout':
    # Pnet = Backbone(input_size, hidden_size, output_size)
    # Enet = Backbone(input_size, hidden_size*2, output_size)

    # Pnet = Backbone_sep(hidden_size,dropout_rate)
    # Enet = Backbone_sep(hidden_size,dropout_rate)

    Pnet = Backbone_sep_V1(hidden_size,dropout_rate=dropout_rate,
                        feature_dim=2)
    Enet = Backbone_sep_V1(hidden_size,dropout_rate=dropout_rate,
                        feature_dim=2)
    Jnet = Joint_net(Pnet, Enet)
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



total_params = sum(p.numel() for p in Jnet.parameters())
print(f"Total number of parameters in Jnet: {total_params}")
#%%
'''
The training part
'''
import torch.optim as optim
# VT grid
GRIDNUM = 20
Vspan = np.linspace(np.min(data_normalized[:,0]), 
                    np.max(data_normalized[:,0]), GRIDNUM)
# Vspan = np.linspace(Vmin_both, Vmax_both, 20)

Tspan = np.linspace(np.min(data_normalized[:,1]), 
                    np.max(data_normalized[:,1]), GRIDNUM)

VV, TT = np.meshgrid(Vspan, Tspan)
VT_grid = np.hstack((VV.reshape(-1,1), TT.reshape(-1,1)))
VT_grid = torch.from_numpy(VT_grid).float().to(device)
VT_grid.requires_grad = True # for computing the Jacobian

# NOTE: add artificial noise to both input and output 
noise_level = 0.05

XX_noise = []
yy_noise = []
VP_H_noise = []
for i in range(10):
    XX_noise.append(np.random.normal(X, np.abs(noise_level*X)))
    yy_noise.append(np.random.normal(y,np.abs(noise_level*y)))
    VP_H_noise.append(np.random.normal(VP_H_normalized, 
                                     np.abs(noise_level/2*VP_H_normalized)) ) 

XX_noise = np.vstack(XX_noise)
yy_noise = np.vstack(yy_noise)
VP_H_noise = np.vstack(VP_H_noise)

EH_noise = E_along_H(VP_H_noise[:,0]*Vscale_both+Vmin_both,
                     VP_H_noise[:,1]*Pscale_both+Pmin_both,
                     V0=5.6648, P0=12, E0 = -8.9432)


XX_noise = torch.tensor(XX_noise).to(device, dtype=torch.float32)
yy_noise = torch.tensor(yy_noise).to(device, dtype=torch.float32)


XX = torch.tensor(X).to(device, dtype=torch.float32)
yy = torch.tensor(y).to(device, dtype=torch.float32)

# XX = torch.normal(XX, XX*noise_level).to(device, dtype=torch.float32)
# yy = torch.normal(XX, XX*noise_level).to(device, dtype=torch.float32)

# VP_hugo = torch.tensor(VP_H).to(device, dtype=torch.float32)
# E_hugo = torch.tensor(E_H[:,None]*E_SCALAR).to(device, dtype=torch.float32)

# NOTE: sharmar used meanH_unscaled = (meanE + 8.9431) + 0.5 * (X_testP[:,0] - 5.6648) * (meanP + 12)*((1e-2)/1.6)
VP_hugo = torch.tensor(VP_H_normalized).to(device, dtype=torch.float32)
E_hugo = torch.tensor(E_H[:,None]).to(device, dtype=torch.float32)

VP_hugo_noise = torch.tensor(VP_H_noise).to(device, dtype=torch.float32)
E_hugo_noise = torch.tensor(EH_noise[:,None]).to(device, dtype=torch.float32)

dT = 1e-3
dV = 1e-3

delta_T = torch.tensor([[0, dT]]).to(device)
delta_V = torch.tensor([[dV, 0]]).to(device)

#%%
'''
Plot the data
'''
fig, ax = plt.subplots(1,2)
ax[0].plot(X[:,0]*Vscale_both+Vmin_both, 
           X[:,1]*Tscale+Tmin, 
           'd')
ax[0].plot(XX_noise[:,0]*Vscale_both+Vmin_both, 
           XX_noise[:,1]*Tscale+Tmin,  '^', alpha=0.5)
ax[0].set_xlabel(r'Volume ($\AA^{3}/atom$)')
ax[0].set_ylabel('Temperature (K)')

ax[1].plot(X[:,0]*Vscale_both+Vmin_both, 
           y*Pscale_both+Pmin_both,
           'd')
ax[1].plot(XX_noise[:,0]*Vscale_both+Vmin_both, 
           yy_noise*Pscale_both+Pmin_both, 
           '^',alpha=0.3)
ax[1].set_xlabel(r'Volume ($\AA^{3}/atom$)')
ax[1].set_ylabel('Pressure (GPa)')
ax[1].legend(['Original', 'Noisy'])
#%%
'''
training loop
'''
# MSEloss = nn.MSELoss()
MSEloss = nn.MSELoss(reduction='mean')
MAEloss = nn.L1Loss()

learning_rate = 5e-4
J_optimizer = optim.Adam(Jnet.parameters(), lr=learning_rate)
# P_optimizer = optim.Adam(Jnet.pnet.parameters(), lr=learning_rate)
# E_optimizer = optim.Adam(Jnet.enet.parameters(), lr=learning_rate)

epochs = 10000

if TRAIN:
    loss_history = []
    for epoch in range(epochs):
        # for i, (X, y) in enumerate(train_loader):
        J_optimizer.zero_grad()
        # P_pred = Jnet.pnet(XX)
        # XX_noisy = torch.normal(XX, 0.01) # sample from a prior normal distribution
        # XX = torch.normal(XX, abs(XX*noise_level)) # use weights instead of augmentation?
        # yy = torch.normal(yy, abs(yy*noise_level))

        # VP_hugo = torch.normal(VP_hugo, VP_hugo*noise_level)
        # E_hugo = torch.normal(E_hugo, E_hugo*noise_level)
        P_pred = Jnet.pnet(XX_noise)   
        E_pred = Jnet.enet(VP_hugo_noise)
        
        E_grid, P_grid = Jnet(VT_grid) # possible to use interpolation with grid data

        if OUT_MODE == 'prob':
            loss_P = -P_pred.log_prob(yy_noise).mean()
            loss_E = -E_pred.log_prob(E_hugo_noise).mean()

            # loss_P = MSEloss(P_pred.mean, yy)
            # loss_E = MSEloss(E_pred.mean, E_hugo)

            # loss_P = torch.exp(-0.5*P_pred.log_prob(yy).mean())
            # loss_E = torch.exp(-0.5*E_pred.log_prob(E_hugo).mean())
            
            # loss_P = ( MSEloss(P_pred.mean, yy) * torch.exp(- P_pred.stddev/P_pred.stddev.sum()) ).sum()
            # loss_E = ( MSEloss(E_pred.mean, E_hugo) * torch.exp(- E_pred.stddev/E_pred.stddev.sum()) ).sum()
            # loss_P = -P_pred.log_prob(yy).mean() + MSEloss(P_pred.mean, yy)
            # loss_E = -E_pred.log_prob(E_hugo).mean() + MSEloss(E_pred.mean, E_hugo)
            # Compute the Jacobian info on regular grid
            # NOTE: The Jacobian can also be computed using finite difference
            
            pE = torch.autograd.grad(E_grid.mean, VT_grid, 
                                    retain_graph = True,
                                    create_graph = True,
                                    grad_outputs = torch.ones_like(E_grid.mean),
                                    allow_unused = False)[0]
            
            pP = torch.autograd.grad(P_grid.mean, VT_grid, 
                                    retain_graph = True,
                                    create_graph = True,
                                    grad_outputs = torch.ones_like(P_grid.mean),
                                    allow_unused = False)[0]
            
            # sample = torch.normal(torch.zeros((VT_grid.shape[0], 1)), 
            #                       torch.ones((VT_grid.shape[0], 1)))
            # pE = torch.autograd.grad(E_grid.mean + E_grid.stddev*sample, 
            #                          VT_grid, 
            #                          retain_graph = True,
            #                          create_graph = True,
            #                          grad_outputs = torch.ones_like(E_grid.mean),
            #                          allow_unused = False)[0]
            
            # pP = torch.autograd.grad(P_grid.mean + P_grid.stddev*sample, 
            #                          VT_grid, 
            #                          retain_graph = True,
            #                          create_graph = True,
            #                          grad_outputs = torch.ones_like(P_grid.mean),
            #                          allow_unused = False)[0]       
        else:
            loss_P = MSEloss(P_pred, yy)
            loss_E = MSEloss(E_pred, E_hugo)

            # loss_P = (MSEloss(P_pred, yy) * torch.exp()).sum()

            # Compute the Jacobian info on regular grid
            # NOTE: The Jacobian can also be computed using finite difference

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
        # loss_D = MSEloss(P_grid*Pscale+Pmin, 
        #                  (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale 
        #                  - 1.6022e-2*pEpV/Vscale)
        loss_C = MSEloss( -pEpV, torch.abs(pEpV)) \
                + MSEloss(pPpT, torch.abs(pPpT))  \
                + MSEloss(pEpT, torch.abs(pEpT))  \
                + MSEloss(-pPpV, torch.abs(pPpV))
        
        # loss_D = MAEloss(P_grid.mean*Pscale_both+Pmin_both, 
        #             (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
        #                 - 1.6022e-2*pEpV/Vscale_both)

        #--------------------------------
        # homogeneous loss-- zero order
        #--------------------------------
        # random_number = torch.randn(1).to(device)
        # E_grid_mul, P_grid_mul = Jnet(random_number*VT_grid)
        # loss_C = MSEloss(E_grid, E_grid_mul) + MSEloss(P_grid, P_grid_mul)
        if EP_MODE == 'variational':
            # if epoch < 4000:
            #     loss = loss_P + loss_E + 0*loss_C + 1*loss_kl
            # else:
            loss_kl = get_kl_loss(Jnet.pnet) + get_kl_loss(Jnet.enet)

            loss = loss_P + loss_E + 1e-3*loss_C + 1e-5*loss_kl
            loss.backward()
            J_optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}, Loss_KL: {loss_kl.item():.4f}')
            

        else:
            loss = loss_P + loss_E + 1e-5*loss_C
            loss.backward()
            J_optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss_P: {loss_P.item():.4f}, Loss_E: {loss_E.item():.4f}, Loss_C: {loss_C.item():.4f}')
        
        loss_history.append(loss.item())
        if OUT_MODE == 'prob':
            if loss_P < -2.0 and loss_E < -1.0:
                break
        else:
            if loss_P < 1e-4 and loss_E < 1e-4:
                break
    plt.plot(loss_history)
    torch.save(Jnet.state_dict(), './weights/temp/exp_joint_uc.pth')


#%%
'''
Check the prediction
NOTE: Load previously trained weights if needed, especially when TRAIN = False
'''
Jnet.load_state_dict(torch.load('./weights/PVT_PV/exp_joint_uc.pth'))
E_pred_sample = []
P_pred_sample = []
E_grid_sample = []
P_grid_sample = []
E_pred_H_sample = []
for i in range(100): # sample 100 times, larger number gives less variations on following results beween each run. But they should be very similar.
    E_pred, P_pred = Jnet(XX)
    E_grid, P_grid = Jnet(VT_grid)
    E_pred_H = Jnet.enet(VP_hugo)
    
    if OUT_MODE == 'prob':
        E_pred = E_pred.mean.cpu().detach().numpy()
        P_pred = P_pred.mean.cpu().detach().numpy()

        E_grid = E_grid.mean.cpu().detach().numpy()
        P_grid = P_grid.mean.cpu().detach().numpy()

        E_pred_H = E_pred_H.mean.cpu().detach().numpy()
    else:
        E_pred = E_pred.cpu().detach().numpy()
        P_pred = P_pred.cpu().detach().numpy()

        E_grid = E_grid.cpu().detach().numpy()
        P_grid = P_grid.cpu().detach().numpy()

        E_pred_H = E_pred_H.cpu().detach().numpy()

    E_pred_sample.append(E_pred)
    P_pred_sample.append(P_pred)
    E_grid_sample.append(E_grid)
    P_grid_sample.append(P_grid)
    E_pred_H_sample.append(E_pred_H)

E_pred_arr = np.array(E_pred_sample)
P_pred_arr = np.array(P_pred_sample)
E_grid_arr = np.array(E_grid_sample)
P_grid_arr = np.array(P_grid_sample)
E_pred_H_arr = np.array(E_pred_H_sample)

'''
if one would like to reproduce the results in the paper, uncomment the following lines
and comment out relevant lines above for direct calculations.
'''
# import pickle as pkl
# with open('./summary/uq_saved_sample.pkl', 'rb') as f:
#     aa = pkl.load(f)
# E_pred, E_pred_std = np.mean(aa['E_X'], axis=0), np.std(aa['E_X'], axis=0)
# P_pred, P_pred_std = np.mean(aa['P_X'], axis=0), np.std(aa['P_X'], axis=0)
# E_grid, E_grid_std = np.mean(aa['E_grid'], axis=0), np.std(aa['E_grid'], axis=0)
# P_grid, P_grid_std = np.mean(aa['P_grid'], axis=0), np.std(aa['P_grid'], axis=0)
# E_pred_H, E_pred_H_std = np.mean(aa['E_H'], axis=0), np.std(aa['E_H'], axis=0)
# pEpV = aa['pEpV']
# pEpT = aa['pEpT']
# pPpV = aa['pPpV']
# pPpT = aa['pPpT']

E_pred, E_pred_std = np.mean(E_pred_arr, axis=0), np.std(E_pred_arr, axis=0)
P_pred, P_pred_std = np.mean(P_pred_arr, axis=0), np.std(P_pred_arr, axis=0)
E_grid, E_grid_std = np.mean(E_grid_arr, axis=0), np.std(E_grid_arr, axis=0)
P_grid, P_grid_std = np.mean(P_grid_arr, axis=0), np.std(P_grid_arr, axis=0)
E_pred_H, E_pred_H_std = np.mean(E_pred_H_arr, axis=0), np.std(E_pred_H_arr, axis=0)

#%%
from sklearn.metrics import r2_score
from scipy.spatial import distance_matrix
from scipy.stats import pearsonr, spearmanr
# Calculate the R2 score for the predicted
Pr2 = r2_score(data[:, 2], P_pred.flatten()*Pscale_both+Pmin_both)
print(f"R2 score (P): {Pr2:.4f}")
P_corr, P_pval = pearsonr(data[:, 2], P_pred.flatten()*Pscale_both+Pmin_both)
print(f"Pearson (P): {P_corr:.4f}, p-val: {P_pval:.2e}")
P_corr_S, P_pval_S = spearmanr(data[:, -1], E_pred[:,0])
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

train_P_err = (P_pred[:,0]*Pscale_both+Pmin_both) - data[:,2]
# train_P_err = (P_pred_interp*Pscale_both+Pmin_both) - data[:,2]
# train_P_err = P_pred.flatten() - data_normalized[:,2]
fig, ax = plt.subplots(2,2)
ax[0,0].plot(data[:,2], train_P_err/data[:,2]*100, '+')
ax[0,0].set_xlabel('Predicted Pressure (GPa)')
ax[0,0].set_ylabel('Error in Pressure %')
ax[0,0].axhline(y=0, color='r', linestyle='--')

ax[0,1].plot(data[:,0], train_P_err/data[:,2]*100, '+')
ax[0,1].set_xlabel('Volume (A^3/atom)')
ax[0,1].set_ylabel('Error in Pressure %')
ax[0,1].axhline(y=0, color='r', linestyle='--')

ax[1,0].plot(data[:,1], train_P_err/data[:,2]*100, '+')
ax[1,0].set_xlabel('Temperature (K)')
ax[1,0].set_ylabel('Error in Pressure %')
ax[1,0].axhline(y=0, color='r', linestyle='--')
plt.tight_layout()


#%%

fig, ax = plt.subplots(2,3)
# Adjust the space between subplots
plt.subplots_adjust(wspace=0.1, hspace=0.1)
# P 
ax[0,0].plot(data[:,2], P_pred.flatten()*Pscale_both+Pmin_both, '+')
ax[0,0].plot(data[:,2],data[:,2], '--', color='r')
# P_idx = np.argsort(P_pred.flatten())
# ax[0,0].fill_between(P_pred[P_idx].flatten()*Pscale_both+Pmin_both,
#                     (P_pred[P_idx]-1.97*P_pred_std[P_idx]).flatten()*Pscale_both+Pmin_both,
#                     (P_pred[P_idx]+1.97*P_pred_std[P_idx]).flatten()*Pscale_both+Pmin_both,
#                     color='pink', alpha=0.4)
ax[0,0].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Pr2,P_corr,P_pval,P_corr_S,P_pval_S),
              transform=ax[0,0].transAxes, fontsize=14, 
              verticalalignment='top')
ax[0,0].set_xlabel('True Pressure (GPa)')
ax[0,0].set_ylabel('Predicted Pressure (GPa)')

ax[1,0].plot(data[:, 2], P_pred.flatten()*Pscale_both+Pmin_both - data[:,-2], '+', label='Error')
ax[1,0].axhline(y=0, color='r', linestyle='--')
# ax[1,0].fill_between(P_pred[P_idx].flatten()*Pscale_both+Pmin_both,
#                     -1.97*P_pred_std[P_idx].flatten()*Pscale_both,
#                     1.97*P_pred_std[P_idx].flatten()*Pscale_both,
#                     color='pink', alpha=0.4)
idx = np.argsort(data[:, 2])
ax[1,0].fill_between(data[idx, 2],
                    -0.05*data[idx, 2],
                    0.05*data[idx, 2],
                    color='pink', alpha=0.4, label='5% error')
ax[1,0].set_xlabel('True Pressure (GPa)')
ax[1,0].set_ylabel('Error in Pressure (GPa)')
ax[1,0].legend()

# E on Hugoniot
ax[0,1].plot(E_H, E_pred_H.flatten(), '+')
ax[0,1].plot(E_H, E_H, '--', color='r')
# EH_idx = np.argsort(E_pred_H.flatten())
# ax[0,1].fill_between(E_pred_H[EH_idx].flatten(),
#                     (E_pred_H[EH_idx]-1.97*E_pred_H_std[EH_idx]).flatten(),
#                     (E_pred_H[EH_idx]+1.97*E_pred_H_std[EH_idx]).flatten(),
#                     color='pink', alpha=0.4)
ax[0,1].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Er2H,E_corr_H,E_pval_H,E_corr_HS,E_pval_HS),
             transform=ax[0,1].transAxes, fontsize=14, 
             verticalalignment='top')
ax[0,1].set_xlabel('True Pressure (GPa) - Hugoniot')
ax[0,1].set_ylabel('Predicted Energy (eV/atom) - Hugoniot')

# ax[1,1].plot(VP_H[:, 1], E_pred_H.flatten()-E_H, '+', label='Error')
ax[1,1].plot(E_H, E_pred_H.flatten()-E_H, '+', label='Error')
idx = np.argsort(E_H)
ax[1,1].fill_between(E_H[idx],
                    -0.01*E_H[idx],
                    0.01*E_H[idx],
                    color='pink', alpha=0.4, label='1% error')
# ax[1,1].fill_between(VP_H[:, 1],
#                     -1.97*E_pred_H_std[EH_idx].flatten(),
#                     1.97*E_pred_H_std[EH_idx].flatten(),
#                     color='pink', alpha=0.4)
# ax[1,1].plot(E_pred_H[EH_idx].flatten(), E_pred_H[EH_idx].flatten()-E_H[EH_idx], '+')
# ax[1,1].fill_between(E_pred_H[EH_idx].flatten(),
#                     -1.97*E_pred_H_std[EH_idx].flatten(),
#                     1.97*E_pred_H_std[EH_idx].flatten(),
#                     color='pink', alpha=0.4)
ax[1,1].axhline(y=0, color='r', linestyle='--')
# ax[1,1].set_xlim([np.min(VP_H[:, 1]), np.max(VP_H[:, 1])])
ax[1,1].set_xlabel('True Energy (eV/atom) - Hugoniot')
ax[1,1].set_ylabel('Error in Energy (eV/atom) - Hugoniot')
ax[1,1].legend()

# E scattered
ax[0,2].plot(data[:,-1], E_pred.flatten(), '+')
ax[0,2].plot(data[:,-1], data[:,-1], '--', color='r')
# E_idx = np.argsort(E_pred.flatten())
# ax[0,2].fill_between(E_pred[E_idx].flatten(),
#                     (E_pred[E_idx]-1.97*E_pred_std[E_idx]).flatten(),
#                     (E_pred[E_idx]+1.97*E_pred_std[E_idx]).flatten(),
#                     color='pink', alpha=0.4)
ax[0,2].text(0.05, 0.95, "$R^2$: {:.02f} \nPearson: {:.02f},\n  pval:{:.2e} \nSpearman: {:.02f},\n  pval:{:.2e}".format(Er2,E_corr_P,E_pval_P,E_corr_S,E_pval_S),
             transform=ax[0,2].transAxes, fontsize=14, 
             verticalalignment='top')
ax[0,2].set_xlabel('True Energy (eV/atom)')
ax[0,2].set_ylabel('Predicted Energy (eV/atom)')

ax[1,2].plot(data[:, -1], E_pred.flatten() - data[:,-1], '+', label='Error')
idx = np.argsort(data[:, -1])
ax[1,2].fill_between(data[idx, -1],
                    -0.3*data[idx, -1],
                    0.3*data[idx, -1],
                    color='pink', alpha=0.4, label='30% error')
# ax[1,2].fill_between(E_pred[E_idx].flatten(),
#                     -1.97*E_pred_std[E_idx].flatten(),
#                     1.97*E_pred_std[E_idx].flatten(),
#                     color='pink', alpha=0.4)
# ax[1,2].plot(data[:, 2], E_pred.flatten() - data[:,-1], '+')
# ax[1,2].fill_between(data[P_idx, 2],
#                     -1.97*E_pred_std[P_idx].flatten(),
#                     1.97*E_pred_std[P_idx].flatten(),
#                     color='pink', alpha=0.4)
ax[1,2].axhline(y=0, color='r', linestyle='--')
ax[1,2].set_xlabel('True Energy (eV/atom)')
ax[1,2].set_ylabel('Error in Energy (eV/atom)')
ax[1,2].legend()

plt.tight_layout()

#%%
'''
A different plot
'''
fig, ax = plt.subplots(1,3, figsize=(10,4))
# ax[0].plot( data[:,2], data[:,0], '.', label='PVT data')
ax[0].errorbar(data[:,2], data[:,0],
               xerr=data[:,2]*0.05, # 3\sigma range
               yerr=data[:,0]*0.05,
               fmt='o', alpha=0.5, label='PVT data')
# ax[0].plot( VP_H[:,1], VP_H[:,0], '^', label='Hugoniot data')
ax[0].errorbar(VP_H[:,1], VP_H[:,0],
               xerr=VP_H[:,1]*0.05, # 3\sigma range
               yerr=VP_H[:,0]*0.05,
               fmt='^', alpha=0.5, label='Hugoniot data')

ax[0].set_xlabel('Pressure (GPa)',size=12)
ax[0].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
ax[0].legend(fontsize=12)

# P 
# ax[1].scatter(data[:,-2], P_pred.flatten()*Pscale_both+Pmin_both,
#               marker='o')
ax[1].errorbar(data[:,2], 
               P_pred.flatten()*Pscale_both+Pmin_both,
               xerr=data[:,2]*0.05,
               yerr=P_pred_std.flatten()*Pscale_both,
               fmt='o', alpha=0.5,linewidth=2)
ax[1].plot(data[:,-2], data[:,-2], '--', color='r')
ax[1].set_xlabel('True Pressure (GPa)',size=12)
ax[1].set_ylabel('Predicted Pressure (GPa)',size=12)

idx = np.argsort(data[:, 2])
ax[1].fill_between(data[idx, 2],
                    0.95*data[idx, 2],
                    1.05*data[idx, 2],
                    color='pink', alpha=0.4, label='5% error')
ax[1].legend(fontsize=12)

# E scattered
# ax[2].scatter(data[:,-1], E_pred.flatten(), marker = 'o')
ax[2].errorbar(data[:,-1], 
               E_pred.flatten(),
               xerr=np.abs(data[:,-1])*0.05,
               yerr=E_pred_std.flatten(),
               fmt='o', alpha=0.5)
# ax[2].scatter(E_H, E_pred_H.flatten(), marker = '^' , 
#               s=100, c='orange')
ax[2].errorbar(E_H, 
               E_pred_H.flatten(),
               xerr=np.abs(E_H)*0.05,
               yerr=E_pred_H_std.flatten(),
               fmt='^', alpha=0.5)
ax[2].plot(data[:,-1], data[:,-1], '--', color='r')
ax[2].set_xlabel('True Energy (eV/atom)',size=12)
ax[2].set_ylabel('Predicted Energy (eV/atom)',size=12)

idx = np.argsort(data[:, -1])
ax[2].fill_between(data[idx, -1],
                    0.75*data[idx, -1],
                    1.25*data[idx, -1],
                    color='pink', alpha=0.4, 
                    label='25% error')
ax[2].set_xlabel('True Energy (eV/atom)',size=12)
ax[2].set_ylabel('Predicted Energy (eV/atom)',size=12)
ax[2].legend(fontsize=12)
plt.tight_layout()

#%%
'''
A different plot (2,2)
'''
fig, ax = plt.subplots(2,2, figsize=(10,8))
# ax[0].plot( data[:,2], data[:,0], '.', label='PVT data')
ax[0][0].errorbar(data[:,2], data[:,0],
               xerr=data[:,2]*0.05, # 3\sigma range
               yerr=data[:,0]*0.05,
               fmt='o', alpha=0.5, label='PVT data')
# ax[0].plot( VP_H[:,1], VP_H[:,0], '^', label='Hugoniot data')
ax[0][0].errorbar(VP_H[:,1], VP_H[:,0],
               xerr=VP_H[:,1]*0.05, # 3\sigma range
               yerr=VP_H[:,0]*0.05,
               fmt='^', alpha=0.5, c='k', ms=12,
               label='Hugoniot data')

ax[0][0].set_xlabel('Pressure (GPa)',size=12)
ax[0][0].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
ax[0][0].legend(fontsize=12)

ax[0][1].errorbar(data[:,1], data[:,0],
               xerr=data[:,1]*0.05, # 3\sigma range
               yerr=data[:,0]*0.05,
               fmt='o', alpha=0.5, label='PVT data')

ax[0][1].set_xlabel('Temperature (K)',size=12)
ax[0][1].set_ylabel(r'Volume ($\AA^{3}/atom$)',size=12)
ax[0][1].legend(fontsize=12)

# P 
# ax[1].scatter(data[:,-2], P_pred.flatten()*Pscale_both+Pmin_both,
#               marker='o')
ax[1][0].errorbar(data[:,2], 
               P_pred.flatten()*Pscale_both+Pmin_both,
               xerr=data[:,2]*0.05,
               yerr=P_pred_std.flatten()*Pscale_both,
               fmt='o', alpha=0.5,linewidth=2)
ax[1][0].plot(data[:,-2], data[:,-2], '--', color='r')
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
ax[1][1].errorbar(data[:,-1], 
               E_pred.flatten(),
               xerr=np.abs(data[:,-1])*0.05,
               yerr=E_pred_std.flatten(),
               fmt='o', alpha=0.5)
# ax[2].scatter(E_H, E_pred_H.flatten(), marker = '^' , 
#               s=100, c='orange')
ax[1][1].errorbar(E_H, 
               E_pred_H.flatten(),
               xerr=np.abs(E_H)*0.05,
               yerr=E_pred_H_std.flatten(), c='k',
               fmt='^', alpha=0.5, ms=12)
ax[1][1].plot(data[:,-1], data[:,-1], '--', color='r')
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
fig, ax = plt.subplots(1,2,figsize=(12, 6))
p0=ax[0].matshow(data[:, -1:] - data[:, -1:].T, cmap='bwr')
cbar = fig.colorbar(p0, ax=ax[0])
cbar.set_label('Energy difference (eV/atom)', 
            rotation=90, fontsize=15, labelpad=12)
cbar.ax.tick_params(labelsize=12)
ax[0].set_title('Energy relative distance matrix (True)')
p1=ax[1].matshow(E_pred - E_pred.T, cmap='bwr')
ax[1].set_title('Energy relative distance matrix (Predicted)')
cbar = fig.colorbar(p1, ax=ax[1])
cbar.set_label('Energy difference (eV/atom)', 
            rotation=90, fontsize=15, labelpad=12)
cbar.ax.tick_params(labelsize=12)

fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(111, projection='3d')
p0 = ax1.scatter(data[:, 0], data[:, 1], 
                 (data[:,-1] - np.min(data[:,-1]))/(np.max(data[:,-1]) - np.min(data[:,-1])), 
                 linewidths=2, alpha=0.5,
                 c='r')
ax1.scatter(data[:, 0], data[:, 1], 
            (E_pred[:,0] - np.min(E_pred[:,0]))/(np.max(E_pred[:,0]) - np.min(E_pred[:,0])),
             linewidths=8, c='b', marker='^')
ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
ax1.set_zlabel('E (normalized)', fontsize=15, labelpad=9)

#%%
# def P_predictor():
#     def func(x):
#         xx = torch.tensor(x).to(device, dtype=torch.float32)
#         p = Jnet.pnet(xx)
#         return p.cpu().detach().numpy()
#     return func
# Tspan = np.linspace(0, 1, 300)
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
# fig = plt.figure(figsize=(18, 12))

# ax1 = fig.add_subplot(221, projection='3d')
# p1 = ax1.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin, 
#                  P_grid[:, 0]*Pscale_both+Pmin_both, 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
#                 # Create a contour plot with P_grid_std values

# contour = ax1.contourf(VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
#                     VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
#                     P_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM))*Pscale_both, 
#                     zdir ='z',
#                     offset = -1,
#                     levels=20, cmap='coolwarm')
# cbar = fig.colorbar(contour, ax=ax1)
# cbar.set_label('Standard Deviation of Pressure (GPa)', 
#             rotation=90, fontsize=15, labelpad=12)
# cbar.ax.tick_params(labelsize=12)
# ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)


# p0 = ax1.scatter(data[:, 0], data[:, 1], data[:,2], linewidths=4,
#                  c='r')
# ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
# ax1.set_zlabel('Pressure (GPa)', fontsize=15, labelpad=9)
# # ax1.legend(['DFT-MD Data'], loc='upper right', fontsize=15)
# # cbar1 = fig.colorbar(p0, ax=ax1, shrink=0.6)
# # cbar1.set_label(r'$\sigma$', rotation=0, fontsize=15, labelpad=12)
# # cbar1.ax.tick_params(labelsize=12)

# ax2 = fig.add_subplot(222, projection='3d')
# p0 = ax2.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin, 
#                  E_grid[:, 0], 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
# p1 = ax2.scatter(data[:, 0], data[:, 1], data[:,-1], 
#                  c='r', linewidths=4)

# contour = ax2.contourf(VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
#                         VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
#                         E_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM)), 
#                         zdir ='z',
#                         offset = -9,
#                         levels=20, cmap='coolwarm')
# ax2.set_zlim(np.min(E_grid)-1, np.max(E_grid)+1)
# cbar = fig.colorbar(contour, ax=ax2)
# cbar.set_label('Standard Deviation of Energy(eV/atom)', 
#             rotation=90, fontsize=15, labelpad=12)
# cbar.ax.tick_params(labelsize=12)
# ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)


# ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
# ax2.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)
# # ax2.legend(['DFT-MD Data'], loc='upper right', fontsize=15)
# # cbar2 = fig.colorbar(p0, ax=ax2, shrink=0.6)
# # cbar2.set_label(r'$\sigma$', rotation=0, fontsize=15, labelpad=12)
# # cbar2.ax.tick_params(labelsize=12)

# ax3 = fig.add_subplot(223, projection='3d')
# p0 = ax3.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
#                  P_grid[:, 0]*Pscale_both+Pmin_both,
#                  E_grid[:, 0], 
#                  alpha=0.5, marker="o", color="k", 
#                  linewidths=4, label="Simulation Data")
# # p1 = ax3.scatter(data[:, 0], data[:, 2], data[:,-1], 
# #                  c='r', linewidths=4)
# p2 = ax3.scatter(VP_H[:,0], VP_H[:,1], E_H, marker='^',
#                  c='r', linewidths=4)
# p2 = ax3.scatter(VP_H[:,0], VP_H[:,1], E_pred_H[:,0],
#                  c='b', linewidths=4)
# ax3.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# ax3.set_ylabel('Pressure (GPa)', fontsize=15, labelpad=12)
# ax3.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)

# # Hugo = E_grid[:, 0] + \
# #        E_SCALAR * 0.5 * (P_grid[:, 0]*Pscale_both+Pmin_both + VP_H[0, 1])\
# #             *(- VP_H[0, 0] + VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both)

# # Hugo = E_grid[:, 0] + 8.9431 +\
# #        E_SCALAR * 0.5 * (P_grid[:, 0]*Pscale_both+Pmin_both + 12)\
# #             *(- 5.6648 + VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both)

# # ax4 = fig.add_subplot(224, projection='3d')
# # p0 = ax4.scatter(VT_grid.detach().numpy()[:, 0]*Vscale_both+Vmin_both, 
# #                 #  VT_grid.detach().numpy()[:, 1]*Tscale+Tmin,
# #                  P_grid[:, 0]*Pscale_both+Pmin_both,
# #                  Hugo, 
# #                  alpha=0.5, marker="o", color="k", 
# #                  linewidths=4, label="Simulation Data")

# # p1 = ax4.scatter(VP_H[:,0], 
# #                  VP_H[:,1],
# #                 #  T_H*Tscale+Tmin,
# #                  np.zeros_like(VP_H[:,0]), 
# #                  alpha=0.5, marker="^", color="r", 
# #                  linewidths=4, label="Exp Data")
# # ax4.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
# # # ax4.set_ylabel('Temperature(K)', fontsize=15, labelpad=12)
# # ax4.set_ylabel('Pressure (GPa)', fontsize=15, labelpad=12)
# # ax4.set_zlabel('Hugoniot (eV/atom)', fontsize=15, labelpad=8)
# plt.tight_layout()
#%%
fig = plt.figure(figsize=(20, 6))
ax1 = fig.add_subplot(131, projection='3d')
# Surface plot
surface = ax1.plot_surface(VV*Vscale_both+Vmin_both, 
                          TT*Tscale+Tmin, 
                          P_grid.reshape((20,20))*Pscale_both+Pmin_both, 
                          alpha=0.4, color='k',
                          linewidth=0,
                          label='Prediction'
                          )
ax1.text2D(0.05, 0.95, "R2 score: {:.04f} \nCorrelation: {:.04f}".format(Pr2,P_corr), 
           transform=ax1.transAxes, fontsize=12, 
           verticalalignment='top')
p0 = ax1.scatter(data[:, 0], data[:, 1], data[:,2], linewidths=4,
                 c='b', label='True')
# contour = ax1.contourf(VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
#                     VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
#                     P_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM))*Pscale_both, 
#                     zdir ='z',
#                     offset = -1,
#                     levels=20, cmap='coolwarm')
# cbar = fig.colorbar(contour, ax=ax1)
# cbar.set_label('Standard Deviation of Pressure (GPa)', 
#             rotation=90, fontsize=15, labelpad=12)
# cbar.ax.tick_params(labelsize=12)
ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
ax1.set_zlabel('Pressure (GPa)', fontsize=15, labelpad=9)
ax1.legend(loc='upper right', fontsize=15)

ax2 = fig.add_subplot(133, projection='3d')
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
# contour = ax2.contourf(VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
#                         VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
#                         E_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM)), 
#                         zdir ='z',
#                         offset = -9,
#                         levels=20, cmap='coolwarm')
# cbar = fig.colorbar(contour, ax=ax2)
# cbar.set_label('Standard Deviation of Pressure (GPa)', 
#             rotation=90, fontsize=15, labelpad=12)
# cbar.ax.tick_params(labelsize=12)
ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
ax2.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)
ax2.legend(loc='upper right', fontsize=15)


ax3 = fig.add_subplot(132, projection='3d')
ax3.plot_surface(VV*Vscale_both+Vmin_both, 
                P_grid[:, 0].reshape((20,20))*Pscale_both+Pmin_both, 
                E_grid.reshape((20,20)), 
                alpha=0.4, color='k',
                linewidth=0,
                label='Prediction'
                )
ax3.text2D(0.05, 0.95, "R2 score: {:.04f} \nCorrelation: {:.04f}".format(Er2H,E_corr_H), 
           transform=ax3.transAxes, fontsize=12, 
           verticalalignment='top')
p1 = ax3.scatter(data[:, 0], data[:, 2], data[:,-1], 
                 c='r', linewidths=4, label='True')
p2 = ax3.scatter(VP_H[:,0], VP_H[:,1], E_H, 
                 c='b', linewidths=4, alpha=0.6, label='True along Hugoniot')
ax3.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax3.set_ylabel('Pressure (GPa)', fontsize=15, labelpad=12)
ax3.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=8)
ax3.legend(loc='upper right', fontsize=15)
plt.tight_layout()

#%%
'''
Contour plots on uncertainty
'''
fig, ax = plt.subplots(1,2)
contour1 = ax[0].contourf(VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
                    VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
                    P_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM))*Pscale_both, 
                    levels=20, cmap='coolwarm')
ax[0].scatter(data[:, 1], data[:, 0], 
              linewidths=P_pred_std[:,0]/np.min(P_pred_std[:,0])*4,
              c='k')
ax[0].set_ylabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax[0].set_xlabel('Temperature (K)', fontsize=15, labelpad=12)
ax[0].set_title('Uncertainty in Pressure')
cbar = fig.colorbar(contour1, ax=ax[0])
cbar.set_label('Standard Deviation of Pressure (GPa)', 
            rotation=90, fontsize=15, labelpad=12)
cbar.ax.tick_params(labelsize=12)
contour2 = ax[1].contourf(VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
                        VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
                        E_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM)), 
                        levels=20, cmap='coolwarm')
ax[1].scatter(data[:, 1], data[:, 0], 
              linewidth=E_pred_std[:,0]/np.min(E_pred_std[:,0])*5,  
              c='k')
ax[1].set_ylabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax[1].set_xlabel('Temperature (K)', fontsize=15, labelpad=12)
ax[1].set_title('Uncertainty in Energy')
cbar = fig.colorbar(contour2, ax=ax[1])
cbar.set_label('Standard Deviation of Energy(eV/atom)', 
            rotation=90, fontsize=15, labelpad=12)
cbar.ax.tick_params(labelsize=12)
# %%
'''
Stitching the plots of interest from previous together
'''
fig, ax = plt.subplots(2,2, figsize=(8,8))
ax[0][0].errorbar(data[:,2], 
               P_pred.flatten()*Pscale_both+Pmin_both,
               xerr=data[:,2]*0.05,
               yerr=P_pred_std.flatten()*Pscale_both,
               fmt='o', alpha=0.5,linewidth=2)
ax[0][0].plot(data[:,-2], data[:,-2], '--', color='r')
ax[0][0].set_xlabel('True Pressure (GPa)',size=15)
ax[0][0].set_ylabel('Predicted Pressure (GPa)',size=15)

idx = np.argsort(data[:, 2])
ax[0][0].fill_between(data[idx, 2],
                    0.95*data[idx, 2],
                    1.05*data[idx, 2],
                    color='pink', alpha=0.4, label='5% error')
ax[0][0].legend(fontsize=12)

# E scattered
# ax[2].scatter(data[:,-1], E_pred.flatten(), marker = 'o')
ax[0][1].errorbar(data[:,-1], 
               E_pred.flatten(),
               xerr=np.abs(data[:,-1])*0.05,
               yerr=E_pred_std.flatten(),
               fmt='o', alpha=0.5, label='Off Hugoniot')
# ax[2].scatter(E_H, E_pred_H.flatten(), marker = '^' , 
#               s=100, c='orange')
ax[0][1].errorbar(E_H, 
               E_pred_H.flatten(),
               xerr=np.abs(E_H)*0.05,
               yerr=E_pred_H_std.flatten(), c='k',
               fmt='^', alpha=0.5, ms=12, label='On Hugoniot')
ax[0][1].plot(data[:,-1], data[:,-1], '--', color='r')
ax[0][1].set_xlabel('True Energy (eV/atom)',size=15)
ax[0][1].set_ylabel('Predicted Energy (eV/atom)',size=15)

idx = np.argsort(data[:, -1])
ax[0][1].fill_between(data[idx, -1],
                    0.75*data[idx, -1],
                    1.25*data[idx, -1],
                    color='pink', alpha=0.4, 
                    label='25% error')
ax[0][1].set_xlabel('True Energy (eV/atom)',size=15)
ax[0][1].set_ylabel('Predicted Energy (eV/atom)',size=15)
ax[0][1].legend(fontsize=12)

# contour plot for uncertainty
contour1 = ax[1][0].contourf(VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
                    VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
                    P_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM))*Pscale_both, 
                    levels=20, cmap='coolwarm')
ax[1][0].scatter(data[:, 1], data[:, 0], 
              linewidths=P_pred_std[:,0]/np.min(P_pred_std[:,0])*4,
              c='k')
ax[1][0].set_ylabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax[1][0].set_xlabel('Temperature (K)', fontsize=15, labelpad=12)
ax[1][0].set_title('Uncertainty in Pressure', fontsize=15)
cbar = fig.colorbar(contour1, ax=ax[1][0],location='bottom', orientation='horizontal')
cbar.set_label('Std.Dev. P (GPa)', 
            rotation=0, fontsize=12, labelpad=12)
cbar.ax.tick_params(labelsize=12)
contour2 = ax[1][1].contourf(VT_grid.detach().numpy()[:, 1].reshape((GRIDNUM,GRIDNUM))*Tscale+Tmin, 
                        VT_grid.detach().numpy()[:, 0].reshape((GRIDNUM,GRIDNUM))*Vscale_both+Vmin_both, 
                        E_grid_std[:, 0].reshape((GRIDNUM,GRIDNUM)), 
                        levels=20, cmap='coolwarm')
ax[1][1].scatter(data[:, 1], data[:, 0], 
              linewidth=E_pred_std[:,0]/np.min(E_pred_std[:,0])*5,  
              c='k')
ax[1][1].set_ylabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
ax[1][1].set_xlabel('Temperature (K)', fontsize=15, labelpad=12)
ax[1][1].set_title('Uncertainty in Energy', fontsize=15)
cbar = fig.colorbar(contour2, ax=ax[1][1], location='bottom', orientation='horizontal')
cbar.set_label('Std.Dev. E(eV/atom)', 
            rotation=0, fontsize=12, labelpad=12)
cbar.ax.tick_params(labelsize=12)

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
    
    if OUT_MODE == 'prob':
        pE = torch.autograd.grad(E_grid.mean, VT_grid, 
                                retain_graph = True,
                                create_graph = True,
                                grad_outputs = torch.ones_like(E_grid.mean),
                                allow_unused = False)[0]
        
        pP = torch.autograd.grad(P_grid.mean, VT_grid, 
                                retain_graph = True,
                                create_graph = True,
                                grad_outputs = torch.ones_like(P_grid.mean),
                                allow_unused = False)[0]
    else:
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
   
    if OUT_MODE == 'prob':
        eqn_err = P_grid.mean*Pscale_both+Pmin_both -  \
                    ( (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
                    - 1.6022e-2*pEpV/Vscale_both)
    else:
        eqn_err = P_grid*Pscale_both+Pmin_both -  \
                    ( (VT_grid[:,1:]+Tmin/Tscale)*pPpT*Pscale_both 
                    - 1.6022e-2*pEpV/Vscale_both)
    
    return eqn_err, pEpV, pEpT, pPpV, pPpT
    

#%%
'''
Comment out this block if one loaded the saved data at line 486
'''
eqn_err = []
pEpV = []
pEpT = []
pPpV = []
pPpT = []
for i in range(100):
    eqn_err_sample, pEpV_sample, pEpT_sample, pPpV_sample, pPpT_sample = forward_pass()
    eqn_err.append( eqn_err_sample.cpu().detach().numpy() )
    pEpV.append( pEpV_sample.cpu().detach().numpy() )
    pEpT.append( pEpT_sample.cpu().detach().numpy() )
    pPpV.append( pPpV_sample.cpu().detach().numpy() )
    pPpT.append( pPpT_sample.cpu().detach().numpy() )

eqn_err = np.array(eqn_err)
pEpV = np.array(pEpV)
pEpT = np.array(pEpT)
pPpV = np.array(pPpV)
pPpT = np.array(pPpT)


#%%
'''
check the emprical density of partial derivatives
'''
fig, ax = plt.subplots(2,2)
ax[0,0].hist(pEpV[...,0].flatten(), bins=20, density='True') # or np.mean(pEpV[...,0],axis=0)?
ax[0,0].set_xlabel(r'$\frac{\partial E}{\partial V}|_T$')
ax[0,1].hist(pEpT[...,0].flatten(), bins=20, density='True')
ax[0,1].set_xlabel(r'$\frac{\partial E}{\partial T}|_V$')
ax[1,0].hist(pPpV[...,0].flatten(), bins=20, density='True')
ax[1,0].set_xlabel(r'$\frac{\partial P}{\partial V}|_T$')
ax[1,1].hist(pPpT[...,0].flatten(), bins=20, density='True')
ax[1,1].set_xlabel(r'$\frac{\partial P}{\partial T}|_V$')
plt.tight_layout()
# %%
