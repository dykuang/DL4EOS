# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# data
path = '../data/VP_sharma.txt'
H_VP = pd.read_csv(path, sep='\s+', header=None)

Sharma_data =  np.loadtxt('../data/data_PVTE.txt')

VP_H = H_VP[1:].to_numpy()
VP_H = VP_H.astype(np.float32)

unique_VP_H, unique_indexes = np.unique(VP_H[:,0], return_index=True)

VP_H = VP_H[unique_indexes]
VP_H = VP_H[::4]

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

E_H = E_along_H(VP_H[:, 0], VP_H[:, 1], 
                V0=5.6648, P0=12, E0 = -8.9432)


def ElasticProperties_density(X, gamma0, q):
    V, P = X[:, 0], X[:, 1]
    gamma = gamma0 * (V / Vr) ** q
    # 1 m^3 Pa/Joule = 10^30 A^3 10^(-9) GPa/(6.2415*10^18 eV) = 1000/6.2415 A^3 GPa/eV
    gamma *= 1000 / 6.2415
    return (P - Pr) * (V / gamma) + Er

Vr = 5.6648
Pr = 12
Er = -8.9431

Xtest = Sharma_data[:,[0,2]]
ytest = Sharma_data[:,-1]

Xtrain = np.vstack((VP_H, Xtest))
ytrain = np.concatenate((E_H, ytest))


popt, pcov = curve_fit(ElasticProperties_density, Xtrain, ytrain, maxfev =10000,  ftol=1e-5, xtol=1e-5, gtol=1e-5)
y_pred = ElasticProperties_density(Xtest, *popt)

#%%
fig = plt.figure(figsize=(10, 8),dpi=150)
ax = fig.add_subplot(111)
p0 = ax.scatter(Sharma_data[:,-1], 
               y_pred,
               marker='o',label = 'Off Hugoniot')
p1 = ax.scatter(E_H, 
               E_H.flatten(),
               marker='^', c= 'k', s=100,label = 'On Hugoniot')
ax.plot(Sharma_data[:,-1], Sharma_data[:,-1], '--', color='r', alpha=0.5)
ax.set_xlabel('True Energy (eV/atom)',size=12)
ax.set_ylabel('Predicted Energy (eV/atom)',size=12)

idx = np.argsort(Sharma_data[:, -1])
ax.fill_between(Sharma_data[idx, -1],
                    0.75*Sharma_data[idx, -1],
                    1.25*Sharma_data[idx, -1],
                    color='pink', alpha=0.4, 
                    label='25% error')
ax.set_xlabel('True Energy (eV/atom)',size=12)
ax.set_ylabel('Predicted Energy (eV/atom)',size=12)
ax.legend(fontsize=12)
plt.tight_layout()

R2 = r2_score(ytest, y_pred)
print("R^2:", R2)

rho_p, p_value_p = pearsonr(ytest, y_pred)
print("Pearson coefficient:", rho_p)
print("p-value:", p_value_p)

rho_s, p_value_s = spearmanr(ytest, y_pred)
print("Spearman coefficient:", rho_s)
print("p-value:", p_value_s)

MSE = np.mean((ytest - y_pred)**2)
RMSE = np.sqrt(MSE)
print("RMSE:", RMSE)
print("paramsL", popt)

E_fit = ElasticProperties_density(VP_H, *popt)  # Use fitted parameters to predict pressure

sorted_indices = np.argsort(E_fit)
V_sorted = VP_H[:,0][sorted_indices]
P_sorted = VP_H[:,1][sorted_indices]
E_sorted = E_fit[sorted_indices]

# plot P-E
plt.figure(figsize=(12, 6), dpi=150)
plt.subplot(1, 2, 1)  
plt.scatter(Xtrain[:,1], ytrain, color="k", label="Train Data", alpha=0.7)
plt.scatter(Xtest[:, 1], ytest, color="b", label="Test Data", alpha=0.7)
plt.scatter(Xtest[:, 1], y_pred, color="r", label="Prediction", marker='^', alpha=0.7)
plt.scatter(P_sorted, E_sorted, color='g', label='Fitted Curve', linewidth=2)  # Plot the fit curve
plt.xlabel('Pressure (GPa)', fontsize=15)
plt.ylabel('Energy (eV/atom)', fontsize=15)
plt.legend(loc="best", fontsize=12)
plt.tight_layout()

# V-E
plt.subplot(1, 2, 2)  
plt.scatter(Xtrain[:,0], ytrain, color="k", alpha=0.7)
plt.scatter(Xtest[:, 0], ytest, color="b", alpha=0.7)
plt.scatter(Xtest[:, 0], y_pred, color="r", marker='^', alpha=0.7)
plt.scatter(V_sorted, E_sorted, color='g', linewidth=2)  # Plot the fit curve
plt.xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15)
plt.ylabel('Energy (eV/atom)', fontsize=15)
plt.tight_layout()
plt.show()

# %%