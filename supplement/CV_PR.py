#%%
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from eos_code import *
import pickle

n_test = 50

'''
CV - split
'''
data = np.loadtxt('../data/data_PVTE.txt')  # V T P E
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # shuffle=True to shuffle data randomly

CV_PR_record = {}

for degree in range(4):

    CV_PR_P_RMSE = []
    CV_PR_E_RMSE = [] 
    fold = 0

    for (train_index, test_index) in kf.split(data):

        train_data = data[train_index]
        test_data = data[test_index]

        X = train_data[:,0:2]                       # Volume-Temperature              
        y1 = train_data[:,2].reshape(-1,1)          # pressure            
        y2 = train_data[:,3].reshape(-1,1)          # energy

        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        scaler1 = MinMaxScaler()
        y_norm = scaler1.fit_transform(np.hstack((y1,y2)))
        y = np.concatenate((y_norm[:,0], y_norm[:,1]))
        y = y.reshape(-1,1)

        X_test = test_data[:,0:2]  

        scaler3 = MinMaxScaler()
        scaler3.fit(X_test)
        X_test = scaler.transform(X_test)

        mu_regression, mu_test_regression = poly_regression(X, y, X_test, n_test, degree+1)
        y11 = mu_test_regression[:X_test.shape[0]].reshape(-1,1)
        y22 = mu_test_regression[X_test.shape[0]:].reshape(-1,1)

        y_mean = np.hstack((y11,y22))
        y_mean1 = scaler1.inverse_transform(y_mean)

        meanP = y_mean1[:,0]
        meanE = y_mean1[:,1]

        P_RMSE = np.sqrt(np.mean((test_data[:,2] - meanP)**2))
        E_RMSE = np.sqrt(np.mean((test_data[:,3] - meanE)**2))

        fold += 1
        
        CV_PR_P_RMSE.append(P_RMSE)
        CV_PR_E_RMSE.append(E_RMSE)


        '''
        Plot
        '''
        # import matplotlib
        # matplotlib.use('Qt5Agg')
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

    CV_PR_record[f'CV_PR_P_RMSE_d{degree+1}'] = CV_PR_P_RMSE
    CV_PR_record[f'CV_PR_E_RMSE_d{degree+1}'] = CV_PR_E_RMSE

    print(f'CV_PR_P_RMSE_d{degree+1}', CV_PR_P_RMSE)
    print(f'Mean CV_PR_P_RMSE_d{degree+1}', np.mean(CV_PR_P_RMSE))
    print(f'CV_PR_E_RMSE_d{degree+1}', CV_PR_E_RMSE)
    print(f'Mean CV_PR_P_RMSE_d{degree+1}', np.mean(CV_PR_E_RMSE))

with open('./supp_summary/CV_PR_record.pkl', 'wb') as f:
    pickle.dump(CV_PR_record, f)


    
# %%
# with open('./supp_summary/CV_PR_record.pkl', 'rb') as f:
#     CV_record = pickle.load(f)

# CV_PR_P_RMSE_d1 = CV_record['CV_PR_P_RMSE_d1']
# CV_PR_E_RMSE_d1 = CV_record['CV_PR_E_RMSE_d1']

import pickle

CV_GP_record = {}

CV_GP_P_RMSE0 = [54.560 	,64.622 	,114.67, 	45.743, 	76.545 ]
CV_GP_E_RMSE0 = [0.377 ,	0.444 ,	0.722 ,	0.295 ,	0.495 ]
CV_GP_P_RMSE1 = [54.560 	,64.622 	,114.67, 	45.743, 	76.545 ]
CV_GP_E_RMSE1 = [0.377 ,	0.444 ,	0.722 ,	0.295 ,	0.495 ]
CV_GP_P_RMSE2 = [13.718 ,	14.153, 	25.783, 	26.730, 	14.349 ]
CV_GP_E_RMSE2 = [0.086, 	0.128 	,0.148 ,	0.229 ,	0.129 ]
CV_GP_P_RMSE3 = []
CV_GP_E_RMSE3 = []
CV_GP_P_RMSE4 = []
CV_GP_E_RMSE4 = []

CV_GP_record['CV_GP_P_RMSE_d1'] = CV_GP_P_RMSE0
CV_GP_record['CV_GP_E_RMSE_d1'] = CV_GP_E_RMSE0
CV_GP_record['CV_GP_P_RMSE_d1'] = CV_GP_P_RMSE1
CV_GP_record['CV_GP_E_RMSE_d1'] = CV_GP_E_RMSE1
CV_GP_record['CV_GP_P_RMSE_d2'] = CV_GP_P_RMSE2
CV_GP_record['CV_GP_E_RMSE_d2'] = CV_GP_E_RMSE2
CV_GP_record['CV_GP_P_RMSE_d3'] = CV_GP_P_RMSE3
CV_GP_record['CV_GP_E_RMSE_d3'] = CV_GP_E_RMSE3
CV_GP_record['CV_GP_P_RMSE_d4'] = CV_GP_P_RMSE4
CV_GP_record['CV_GP_E_RMSE_d4'] = CV_GP_E_RMSE4