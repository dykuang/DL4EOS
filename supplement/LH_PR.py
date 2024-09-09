#%%
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from eos_code import *

LH = 0
n_test = 50
degree = 3
data = np.loadtxt('../data/data_PVTE.txt')  # V T P E

LH_PR_P_RMSE = []
LH_PR_E_RMSE = [] 
LH_PR_record = {}
'''
LH - split
'''

for i in range(4):
    TYPE = i+1
    if TYPE == 1:
        condition = np.logical_and(data[:, 1] <= 4500, data[:, 2] <= 400)
    if TYPE == 2:
        condition = np.logical_and(data[:, 1] <= 5500, data[:, 2] <= 500)
    if TYPE == 3:
        condition = np.logical_and(data[:, 1] <= 6500, data[:, 2] <= 650)
    if TYPE == 4:
        condition = np.logical_and(data[:, 1] <= 7500, data[:, 2] <= 700)

    train_data = data[condition]
    test_data = data[~condition]

    X = data[:,0:2]                       # Volume-Temperature              
    y1 = data[:,2].reshape(-1,1)          # pressure            
    y2 = data[:,3].reshape(-1,1)          # energy

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_test = X[~condition] 
    X_train = X[condition]

    scaler1 = MinMaxScaler()
    y_norm = scaler1.fit_transform(np.hstack((y1,y2)))
    y_norm = y_norm[condition]

    y = np.concatenate((y_norm[:,0], y_norm[:,1]))
    y = y.reshape(-1,1)

    mu_regression, mu_test_regression = poly_regression(X_train, y, X_test, n_test, degree)
    y11 = mu_test_regression[:X_test.shape[0]].reshape(-1,1)
    y22 = mu_test_regression[X_test.shape[0]:].reshape(-1,1)

    y_mean = np.hstack((y11,y22))
    y_mean1 = scaler1.inverse_transform(y_mean)

    meanP = y_mean1[:,0]
    meanE = y_mean1[:,1]

    P_RMSE = np.sqrt(np.mean((test_data[:,2] - meanP)**2))
    E_RMSE = np.sqrt(np.mean((test_data[:,3] - meanE)**2))

    LH += 1
    
    LH_PR_P_RMSE.append(P_RMSE)
    LH_PR_E_RMSE.append(E_RMSE)


    '''
    Plot
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(18, 8),dpi=150)
    ax1 = fig.add_subplot(121, projection='3d')
    p0 = ax1.scatter(train_data[:, 0], train_data[:, 1], train_data[:, 2], alpha=1, marker="o", color="k",linewidths=4)
    p1 = ax1.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], alpha=1, marker="o", color="b",linewidths=4)
    p2 = ax1.scatter(test_data[:, 0], test_data[:, 1], meanP, marker= '^', color = 'r',linewidths=4)
    ax1.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
    ax1.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
    ax1.set_zlabel('Pressure (GPa)', fontsize=15, labelpad=12)
    

    ax2 = fig.add_subplot(122, projection='3d')
    p0 = ax2.scatter(train_data[:, 0], train_data[:, 1], train_data[:,3], alpha=1, marker="o", color="k", linewidths=4, label="Data")
    p1 = ax2.scatter(test_data[:, 0], test_data[:, 1], test_data[:,3], marker="o", color="b", linewidths=4)
    p2 = ax2.scatter(test_data[:, 0], test_data[:, 1], meanE, alpha=1, marker="^", color="r", linewidths=4)
    ax2.set_xlabel(r'Volume ($\AA^{3}/atom$)', fontsize=15, labelpad=8)
    ax2.set_ylabel('Temperature (K)', fontsize=15, labelpad=12)
    ax2.set_zlabel('Energy (eV/atom)', fontsize=15, labelpad=12)
    ax2.legend(['Train','Test', 'Predict'], loc='upper right', fontsize=15)
    plt.show()   
    '''
    save the LH record
    '''

    LH_PR_record[f'LH_PR_P_RMSE_region{TYPE}'] = P_RMSE
    LH_PR_record[f'LH_PR_E_RMSE_region{TYPE}'] = E_RMSE

    print(f'LH_PR_P_RMSE_region{TYPE}', P_RMSE)
    # print(f'Mean LH_PR_P_RMSE_region{TYPE}', np.mean(P_RMSE))
    print(f'LH_PR_E_RMSE_region{TYPE}', E_RMSE)


with open('./supp_summary/LH_PR_record.pkl', 'wb') as f:
    pickle.dump(LH_PR_record, f)

# %%
# with open('./supp_summary/LH_PR_record.pkl', 'rb') as f:
#     LH_record = pickle.load(f)

# LH_PR_P_RMSE = LH_record['LH_PR_P_RMSE_region1']
# LH_PR_E_RMSE = LH_record['LH_PR_E_RMSE_region1']



