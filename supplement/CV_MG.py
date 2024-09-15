# %%
import numpy as np
import pickle
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

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

# Load data and set up cross-validation
data = np.loadtxt('../data/data_PVTE.txt')    # V(A^3/atom) P(GPa) T(K) E(eV/atom)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

CV_MG_RMSE, CV_MG_params = [], []

for train_idx, test_idx in kf.split(data):
    train_data, test_data = data[train_idx], data[test_idx]
    ytrain = train_data[:, 3]  
    ytest = test_data[:, 3]

    # Fit the model
    popt, pcov = curve_fit(ElasticProperties_density, train_data, ytrain, maxfev=10000, ftol=1e-5, xtol=1e-5, gtol=1e-5)
    ypred = ElasticProperties_density(test_data, *popt)

    RMSE = np.sqrt(np.mean((ytest - ypred) ** 2))
    CV_MG_RMSE.append(RMSE)
    CV_MG_params.append(popt)

    
    E_fit = ElasticProperties_density(train_data, *popt)  
    sorted_indices = np.argsort(E_fit)
    V_sorted = train_data[:,0][sorted_indices]
    P_sorted = train_data[:,2][sorted_indices]
    E_sorted = E_fit[sorted_indices]

    plt.figure(figsize=(12, 6), dpi=150)

    # plot P-E
    plt.subplot(1, 2, 1)  
    plt.scatter(train_data[:, 2], ytrain, color="k", label="Train Data", alpha=0.7)
    plt.scatter(test_data[:, 2], ytest, color="b", label="Test Data", alpha=0.7)
    plt.scatter(test_data[:, 2], ypred, color="r", label="Prediction", marker='^', alpha=0.7)
    plt.scatter(P_sorted, E_sorted, color='g', label='Fitted Curve', linewidth=2)  # Plot the fit curve
    plt.xlabel('Pressure (GPa)', fontsize=15)
    plt.ylabel('Energy (eV/atom)', fontsize=15)
    plt.legend(loc="best", fontsize=12)
    plt.tight_layout()

    # V-E
    plt.subplot(1, 2, 2)  
    plt.scatter(train_data[:, 0], ytrain, color="k", alpha=0.7)
    plt.scatter(test_data[:, 0], ytest, color="b", alpha=0.7)
    plt.scatter(test_data[:, 0], ypred, color="r", marker='^', alpha=0.7)
    plt.scatter(V_sorted, E_sorted, color='g', linewidth=2)  # Plot the fit curve
    plt.xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15)
    plt.ylabel('Energy (eV/atom)', fontsize=15)
    plt.tight_layout()
    plt.show()


# Save results
CV_record = {'CV_MG_RMSE': CV_MG_RMSE, 'CV_MG_params': CV_MG_params}
print("CV_MG_RMSE:", CV_MG_RMSE)
print("Mean CV_MG_RMSE:", np.mean(CV_MG_RMSE))
print("CV_MG_params:", CV_MG_params)
with open('./supp_summary/CV_MG_record.pkl', 'wb') as f:
    pickle.dump(CV_record, f)

# %%