#%%
import numpy as np
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import fmin_cobyla
from sklearn.model_selection import KFold
from eos_code import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_test = 50
CV_GP_record = {}
constraints = 1
'''
CV - split
'''
data = np.loadtxt('../data/data_PVTE.txt')  # V T P E
kf = KFold(n_splits=5, shuffle=True, random_state=42)   # shuffle=True to shuffle data randomly

for degree in range(5):

    CV_GP_P_RMSE = []
    CV_GP_E_RMSE = [] 
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


        X_test,_ = test(X, n_test = 50)                # test points
        X_c,_  = test(X, n_test = 10)                   # virtual points

        mu_regression, mu_test_regression = poly_regression(X, y, X_test, n_test, degree)

        class GP_EOS:
            def __init__(self, X, y, mu_regression, X_c, n_test, degree):
                self.X = X
                self.y = y
                self.mu_regression = mu_regression
                self.X_c = X_c
                self.n_test = n_test
                self.degree = degree

            def neglikelihood(self, theta):
                return mle(theta, self.X, self.y, self.mu_regression, constraints)

            def predict_virtual(self, theta):
                return predict_prime(theta, self.X, self.y, self.X_c, self.n_test, self.degree, constraints)

            def nonneg_const(self, theta):
                yy, s = self.predict_virtual(theta)
                y_prime_mean_P = yy[:self.X_c.shape[0]]
                y_prime_std_P = s[:self.X_c.shape[0]]

                y_prime_mean_E = yy[self.X_c.shape[0]:]
                y_prime_std_E = s[self.X_c.shape[0]:]

                const1 = y_prime_mean_P + 1.96 * np.sqrt(y_prime_std_P)
                const2 = y_prime_mean_E - 1.96 * np.sqrt(y_prime_std_E)

                const = np.concatenate((-const1, const2))

                return const

            def optimize(self, constraints):
                if constraints == 1:
                    cons = self.nonneg_const
                else:
                    cons = []

                n = 10
                lb = np.array([-3.0, -3.0, -3.0, -10.0, -10.0])
                r = np.random.standard_normal(size=(n, 5))
                x0 = r + lb
                optipar = np.zeros((n, 5))
                optifun = np.zeros((n, 1))

                for i in range(n):
                    print(i)
                    res = fmin_cobyla(self.neglikelihood, x0[i, :].T, cons=cons, args=(),
                                    consargs=None, rhobeg=1.0, rhoend=0.0001, maxfun=1000,
                                    disp=True, catol=0.0002)
                    optipar[i, :] = res
                    print(optipar[i, :])
                    optifun[i, :] = self.neglikelihood(res)
                    print(optifun[i, :])

                theta = optipar[np.argmin(optifun)]

                return theta
        optimizer = GP_EOS(X, y, mu_regression, X_c, n_test, degree)
        thetaopt = optimizer.optimize(constraints)

        X_test = test_data[:,0:2]  
        scaler3 = MinMaxScaler()
        scaler3.fit(X_test)
        X_test = scaler.transform(X_test)

        y_pred, y_std = predict(thetaopt, X, y, X_test, n_test, degree, constraints)

        y11 = y_pred[:X_test.shape[0]].reshape(-1,1)
        y22 = y_pred[X_test.shape[0]:].reshape(-1,1)

        y_mean = np.hstack((y11,y22))
        y_mean1 = scaler1.inverse_transform(y_mean)

        meanP = y_mean1[:,0]
        meanE = y_mean1[:,1]

        P_mse = np.mean((test_data[:,2] - meanP)**2)
        E_mse = np.mean((test_data[:,3] - meanE)**2)

        P_RMSE = np.sqrt(np.mean((test_data[:,2] - meanP)**2))
        E_RMSE = np.sqrt(np.mean((test_data[:,3] - meanE)**2))

        fold += 1
        
        CV_GP_P_RMSE.append(P_RMSE)
        CV_GP_E_RMSE.append(E_RMSE)
        '''
        Plot
        '''

        
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

    CV_GP_record[f'CV_GP_P_RMSE_d{degree}'] = CV_GP_P_RMSE
    CV_GP_record[f'CV_GP_E_RMSE_d{degree}'] = CV_GP_E_RMSE

    print(f'CV_GP_P_RMSE_d{degree}', CV_GP_P_RMSE)
    print(f'Mean CV_GP_P_RMSE_d{degree}', np.mean(CV_GP_P_RMSE))
    print(f'CV_GP_E_RMSE_d{degree}', CV_GP_E_RMSE)
    print(f'Mean CV_GP_P_RMSE_d{degree}', np.mean(CV_GP_E_RMSE))

with open('./supp_summary/CV_GP_record.pkl', 'wb') as f:
    pickle.dump(CV_GP_record, f)
# %%
# with open('./supp_summary/CV_GP_record.pkl', 'rb') as f:
#     CV_record = pickle.load(f)

# CV_GP_P_RMSE_d1 = CV_record['CV_GP_P_RMSE_d1']
# CV_GP_E_RMSE_d1 = CV_record['CV_GP_E_RMSE_d1']