'''
Author @ 2024 Dongyang Kuang
helper functions for convenience
'''
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# More from https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
import matplotlib.pyplot as plt
import numpy as np

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

def E_along_H(V, P, V0=0, P0=0, E0=0):
    '''
    P: 1Gpa = 1e-9 Pa
    V: 1A^3 = 1e-30 m^3
    E: 1eV = 1.6022e-19 J
    using formular E - E0 = 0.5(P+P0)(V0-V)
    return
        E - E0 (in eV)
   
    # NOTE: sharmar used 
    meanH_unscaled = (meanE + 8.9431) + 0.5 * (X_testP[:,0] - 5.6648) * (meanP + 12)*((1e-2)/1.6)

    '''
    return 0.5*(P+P0)*(V0-V)*((1e-2)/1.6022) + E0

from scipy.interpolate import CubicSpline
def Get_T_from_PV(P_predictor, V, Tspan, P_query):
    '''
    Get T from P and V by linear interpolation
    input:
        P_predictor: a function that takes in (V,T) and return P
        V: a float number
        Tspan: one dimensional array of temperature
        P_query: a float number at which we want to query the temperature
    return:
        T: a float number
    '''
    VTgrid = np.c_[V*np.ones_like(Tspan), Tspan]
    P_pred = P_predictor(VTgrid).squeeze()
    P_pred_min, P_pred_max = P_pred.min(), P_pred.max()
    sorted_idx = np.argsort(P_pred)
    if P_query < P_pred_min or P_query > P_pred_max:
        raise ValueError(f'Queried point is out of range [{P_pred_min}, {P_pred_max}], {P_query}. ')
    else:
        # cs = CubicSpline(P_pred[sorted_idx], Tspan[sorted_idx])
        # T = cs(P_query)
        T = np.interp(P_query, P_pred[sorted_idx], Tspan[sorted_idx])
    
    return T


#%%
# if __name__ == '__main__':
#     '''
#     Check on Sharma's data
#     '''
#     from matplotlib.patches import Rectangle
#     path = './data/VP_sharma.txt'
#     H_VP = pd.read_csv(path, sep='\s+', header=None)
#     # data = np.loadtxt(path)
#     da = H_VP[3:].to_numpy()
#     da = da.astype(float)
#     E_H = E_along_H(da[:, 0],da[:, 1],  V0=da[0, 0], P0=da[0, 1])
    
#     # %%
#     fig = plt.figure()
#     ax = fig.add_subplot(121)
#     ax.scatter(da[:, 0], da[:, 1])
#     ax.set_xlabel('V(A^3)')
#     ax.set_ylabel('P(GPa)')

#     ax1 = fig.add_subplot(122, projection='3d')
#     ax1.scatter(da[:, 0], da[:, 1], E_H)

#     ax1.set_xlabel('V(A^3)')
#     ax1.set_ylabel('P(GPa)')
#     ax1.set_zlabel('E - E0')

#     # plt.show()


#     # %%
#     data_pvte = np.loadtxt('./data/data_PVTE.txt')
#     E_SCALAR = 6.241509326*1e-3
#     plt.figure()
#     plt.scatter(data_pvte[:,0], data_pvte[:,-1], label='Static') # V, E
#     plt.scatter(da[:, 0], E_H*E_SCALAR, label = 'E along H', color = 'red')
#     plt.legend()
#     plt.xlabel('V(A^3)')
#     plt.ylabel('E')

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data_pvte[:,0], data_pvte[:,2], data_pvte[:,-1]) # V, E
#     ax.scatter(da[:, 0], da[:,1], E_H*E_SCALAR,c='r')
#     # plt.legend()
#     plt.xlabel('V(A^3)')
#     plt.ylabel('P(GPa)') 

#     plt.figure()
#     plt.scatter(data_pvte[:,0], data_pvte[:,2], label='Static') # V, P
#     plt.scatter(da[:, 0], da[:,1], label = 'Along H', color = 'red')
#     plt.legend()
#     plt.xlabel('V(A^3)')
#     plt.ylabel('P(GPa)')    


#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data_pvte[:,0], data_pvte[:,2], data_pvte[:,1],
#                s=20) # T on V-P plane
#     plt.xlabel('V(A^3)')
#     plt.ylabel('P(GPa)') 

#     # %%
#     data_scale = np.max(data_pvte, axis=0, keepdims=True) - np.min(data_pvte, axis=0, keepdims=True)
#     # margin_scale = 1.05*data_scale

#     data_normalized = (data_pvte - np.min(data_pvte, axis=0, keepdims=True))/(np.max(data_pvte, axis=0, keepdims=True) - np.min(data_pvte, axis=0, keepdims=True))

#     Vmin, Tmin, Pmin, Emin = np.min(data_pvte, axis=0).astype('float32')
#     Vmax, Tmax, Pmax, Emax = np.max(data_pvte, axis=0).astype('float32')
#     Vscale, Tscale, Pscale, Escale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin, Emax-Emin

#     Vmin_both = min(Vmin, np.min(da[:,0]))
#     Vmax_both = max(Vmax, np.max(da[:,0]))
#     Vscale_both = Vmax_both - Vmin_both

#     Pmin_both = min(Pmin, np.min(da[:,1]))
#     Pmax_both = max(Pmax, np.max(da[:,1]))
#     Pscale_both = Pmax_both - Pmin_both

#     data_normalized[:,0] = (data_pvte[:,0] - Vmin_both)/Vscale_both
#     data_normalized[:,2] = (data_pvte[:,2] - Pmin_both)/Pscale_both

#     X = data_normalized[:,:2].astype(np.float32)
#     y = data_normalized[:,2:3].astype(np.float32)

#     degree = 3
#     model = LinearRegression()
#     reg_model, poly_feature_trans = poly_regression(X, y, degree, model)

#     P_predictor = lambda x: make_prediction(reg_model, poly_feature_trans, x)

#     V_H_normalized = (da[:,0] - Vmin_both)/Vscale_both
#     P_H_normalized = (da[:,1] - Pmin_both)/Pscale_both
    
#     vv = np.linspace(0, 1, 100)
#     tt = np.linspace(0, 1, 100)
#     VV, TT = np.meshgrid(vv, tt)
#     VT_grid = np.c_[VV.ravel(), TT.ravel()]
#     P_grid = P_predictor(VT_grid).reshape(VV.shape)
#     #%%
#     Tspan = np.linspace(0, 1, 100)
#     T = []

#     for i in range(len(da)):
#         Temp = Get_T_from_PV(P_predictor, V_H_normalized[i], Tspan, P_H_normalized[i])
#         print(f'V: {da[i,0]}, P: {da[i,1]}, T: {Temp*Tscale+Tmin}')
#         T.append(Temp)
#     T = np.array(T)
#     # %%
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(data_pvte[:,0], data_pvte[:,1], data_pvte[:,2],
#                 s=20) # T on V-P plane
#     ax.scatter(da[:,0], T*Tscale+Tmin, da[:,1],
#                 s=20) # T on V-P plane    
#     plt.xlabel('V(A^3)')
#     plt.ylabel('T(K)') 
    
#     Plevels = np.linspace(np.min(P_grid*Pscale_both+Pmin_both), 
#                          np.max(P_grid*Pscale_both+Pmin_both), 20)
#     Vlevels = np.linspace(np.min(VV*Vscale_both+Vmin_both), 
#                          np.max(VV*Vscale_both+Vmin_both), 20)
#     fig, ax = plt.subplots(1,3)
#     ax[0].contourf(VV*Vscale_both+Vmin_both,
#                    TT*Tscale+Tmin,
#                    P_grid*Pscale_both+Pmin_both,
#                   levels=Plevels, cmap='viridis')
#     ax[0].scatter(da[:,0], T*Tscale+Tmin,marker='+', c='r')
#     ax[0].add_patch(Rectangle((Vmin,Tmin),Vscale,Tscale,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     # ax[0].imshow(P_grid*Pscale+Pmin)
#     ax[0].set_xlabel('V(A^3)')
#     ax[0].set_ylabel('T(K)')

#     ax[1].contourf(P_grid*Pscale_both+Pmin_both,
#                    TT*Tscale+Tmin, 
#                    VV*Vscale_both+Vmin_both,
#                   levels=Vlevels, cmap='viridis')
#     ax[1].scatter(da[:,1], T*Tscale+Tmin,marker='+', c='r')
#     ax[1].add_patch(Rectangle((Pmin,Tmin),Pscale,Tscale,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     ax[1].set_xlabel('P(GPa)')
#     ax[1].set_ylabel('T(K)')

#     ax[2].contourf(VV*Vscale_both+Vmin_both, 
#                    P_grid*Pscale_both+Pmin_both, 
#                    TT*Tscale+Tmin)
#     ax[2].add_patch(Rectangle((Vmin,Pmin),Vscale,Pscale,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     ax[2].scatter(da[:,0], da[:,1],marker='+', c='r')
#     ax[2].set_xlabel('V(A^3)')
#     ax[2].set_ylabel('P(GPa)')

#     # %%
#     '''
#     check on the exp hugoniot data
#     '''
#     VTP_df = pd.read_csv(r'P-T-V-Fe.txt', sep='\t')
#     VTP_da = VTP_df[['V(angstrom3)', 'T(K)', 'P(GPa)']].to_numpy()

#     exp_H_PV = np.loadtxt('Hugoniont.txt') #(P, V)
#     exp_H_PV[:,1] = 558.45/6.022 * exp_H_PV[:,1] * 2  # convert from cm^3/gto A^3/2 atom
#     exp_H_VP = np.c_[exp_H_PV[:,1], exp_H_PV[:,0]] # (V, P) 

#     E_H = E_along_H(exp_H_VP[:, 0], exp_H_VP[:, 1], 
#                     V0=exp_H_VP[0, 0], P0=exp_H_VP[0, 1])
#     E_H = E_H * E_SCALAR

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(exp_H_VP[:, 0], exp_H_VP[:,1], E_H,c='r')
#     # plt.legend()
#     plt.xlabel('V(A^3)')
#     plt.ylabel('P(GPa)') 

#     # %%
#     data_scale = np.max(VTP_da, axis=0, keepdims=True) - np.min(VTP_da, axis=0, keepdims=True)
#     # margin_scale = 1.05*data_scale
#     Vmin, Tmin, Pmin = np.min(VTP_da, axis=0).astype('float32')
#     Vmax, Tmax, Pmax = np.max(VTP_da, axis=0).astype('float32')
#     Vscale, Tscale, Pscale = Vmax-Vmin, Tmax-Tmin, Pmax-Pmin

#     data_normalized = (VTP_da - np.min(VTP_da, axis=0, keepdims=True))/(np.max(VTP_da, axis=0, keepdims=True) - np.min(VTP_da, axis=0, keepdims=True))
    
#     Vmin_both = min(Vmin, np.min(exp_H_VP[:,0]))
#     Vmax_both = max(Vmax, np.max(exp_H_VP[:,0]))
#     Vscale_both = Vmax_both - Vmin_both

#     Pmin_both = min(Pmin, np.min(exp_H_VP[:,1]))
#     Pmax_both = max(Pmax, np.max(exp_H_VP[:,1]))
#     Pscale_both = Pmax_both - Pmin_both

#     data_normalized[:,0] = (VTP_da[:,0] - Vmin_both)/Vscale_both
#     data_normalized[:,2] = (VTP_da[:,2] - Pmin_both)/Pscale_both

#     X = data_normalized[:,:2].astype(np.float32)
#     y = data_normalized[:,2:3].astype(np.float32)

#     degree = 3
#     model = LinearRegression()
#     reg_model, poly_feature_trans = poly_regression(X, y, degree, model)

#     P_predictor = lambda x: make_prediction(reg_model, poly_feature_trans, x)

#     V_H_normalized = (exp_H_VP[:,0] - Vmin_both)/Vscale_both
#     P_H_normalized = (exp_H_VP[:,1] - Pmin_both)/Pscale_both
    
#     vv = np.linspace(0, 1, 100)
#     tt = np.linspace(0, 2, 200)
#     VV, TT = np.meshgrid(vv, tt)
#     VT_grid = np.c_[VV.ravel(), TT.ravel()]
#     P_grid = P_predictor(VT_grid).reshape(VV.shape)

#     # %%
#     Tspan = np.linspace(0, 1.5, 150)
#     T = []

#     for i in range(len(exp_H_VP)):
#         Temp = Get_T_from_PV(P_predictor, V_H_normalized[i], 
#                              Tspan, P_H_normalized[i])
#         print(f'V: {exp_H_VP[i,0]}, P: {exp_H_VP[i,1]}, T: {Temp*Tscale+Tmin}')
#         T.append(Temp)
#     T = np.array(T)

#     # %%
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(VTP_da[:,0], VTP_da[:,1], VTP_da[:,2],
#                 s=20) # T on V-P plane
#     ax.scatter(exp_H_VP[:,0], T*Tscale+Tmin, exp_H_VP[:,1],
#                 s=20) # T on V-P plane    
#     plt.xlabel('V(A^3)/2')
#     plt.ylabel('T(K)') 
    
#     Plevels = np.linspace(np.min(P_grid*Pscale_both+Pmin_both), 
#                          np.max(P_grid*Pscale_both+Pmin_both), 20)
#     Vlevels = np.linspace(np.min(VV*Vscale_both+Vmin_both), 
#                          np.max(VV*Vscale_both+Vmin_both), 20)
#     fig, ax = plt.subplots(1,3,figsize=(12,6))
#     ax[0].contourf(VV*Vscale_both+Vmin_both,
#                    TT*Tscale+Tmin,
#                    P_grid*Pscale_both+Pmin_both,
#                   levels=Plevels, cmap='viridis')
#     ax[0].scatter(exp_H_VP[:,0], T*Tscale+Tmin,marker='+', c='r')
#     # ax[0].imshow(P_grid*Pscale+Pmin)
#     ax[0].add_patch(Rectangle((Vmin_both,Tmin),Vscale_both,Tscale,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     ax[0].set_xlabel('V(A^3)')
#     ax[0].set_ylabel('T(K)')
    
#     ax[1].contourf(P_grid*Pscale_both+Pmin_both,
#                    TT*Tscale+Tmin, 
#                    VV*Vscale_both+Vmin_both,
#                   levels=Vlevels, cmap='viridis')
#     ax[1].scatter(exp_H_VP[:,1], T*Tscale+Tmin,marker='+', c='r')
#     ax[1].add_patch(Rectangle((Pmin_both,Tmin),Pscale_both,Tscale,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     ax[1].set_xlabel('P(GPa)')
#     ax[1].set_ylabel('T(K)')

#     ax[2].contourf(VV*Vscale_both+Vmin_both, 
#                    P_grid*Pscale_both+Pmin_both, 
#                    TT*Tscale+Tmin)
#     ax[2].scatter(exp_H_VP[:,0], exp_H_VP[:,1],marker='+', c='r')
#     ax[2].add_patch(Rectangle((Vmin_both,Pmin_both),
#                               Vscale_both,Pscale_both,
#                     edgecolor='red',
#                     facecolor='none',
#                     lw=2))
#     ax[2].set_xlabel('V(A^3)')
#     ax[2].set_ylabel('P(GPa)')
# %%
