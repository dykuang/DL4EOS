#%%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Constants from Sharma et al.2024
Vr = 5.6648
Pr = 12
Er = -8.9431

# Define model function
def ElasticProperties_density(X, gamma0, q):
    V, P = X[:, 0], X[:, 2]
    gamma = gamma0 * (V / Vr) ** q
    # 1 m^3 Pa/Joule = 10^30 A^3 10^(-9) GPa/(6.2415*10^18 eV) = 1000/6.2415 A^3 GPa/eV
    gamma *= 1000 / 6.2415
    return (P - Pr) * (V / gamma) + Er

# Load data
data = np.loadtxt('../data/data_PVTE.txt')
conditions = [
    (data[:, 1] <= 4500) & (data[:, 2] <= 400),
    (data[:, 1] <= 5500) & (data[:, 2] <= 500),
    (data[:, 1] <= 6500) & (data[:, 2] <= 650),
    (data[:, 1] <= 7500) & (data[:, 2] <= 700)
]

LH_MG_RMSE, LH_MG_params = [], []

# Loop over conditions and perform curve fitting
for condition in conditions:
    train_data, test_data = data[condition], data[~condition]
    Xtrain, ytrain = train_data[:, :3], train_data[:, 3]
    Xtest, ytest = test_data[:, :3], test_data[:, 3]

    popt, pcov = curve_fit(ElasticProperties_density, Xtrain, ytrain, maxfev=10000, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    ypred = ElasticProperties_density(Xtest, *popt)

    LH_MG_RMSE.append(np.sqrt(np.mean((ytest - ypred) ** 2)))
    LH_MG_params.append(popt)

    # Plot results
    fig = plt.figure(figsize=(12, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 3], color="k", marker="o", linewidths=4)
    ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 3], color="b", marker="o", linewidths=4)
    ax.scatter(test_data[:, 0], test_data[:, 1], ypred, color="r", marker="^", linewidths=4)

    ax.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15)
    ax.set_ylabel('Temperature (K)', fontsize=15)
    ax.set_zlabel('Energy (eV/atom)', fontsize=15)
    plt.tight_layout()
    plt.show()

# Save results
LH_record = {'LH_MG_RMSE': LH_MG_RMSE, 'LH_MG_params': LH_MG_params}
print("LH_MG_RMSE:", LH_MG_RMSE)
print("Mean LH_MG_RMSE:", np.mean(LH_MG_RMSE))

with open('./supp_summary/LH_MG_record.pkl', 'wb') as f:
    pickle.dump(LH_record, f)

# %%
# with open('./supp_summary/LH_MG_record.pkl', 'rb') as f:
#     LH_MG_record = pickle.load(f)

# LH_MG_RMSE = LH_record['LH_MG_RMSE']
# LH_MG_params = LH_record['LH_MG_params']