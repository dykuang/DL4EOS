# %%
import numpy as np
from matplotlib.patches import Rectangle
import pickle
import matplotlib.pyplot as plt

data = np.loadtxt('../data/data_PVTE.txt')  # V T P E

with open('../summary/LH_record.pkl', 'rb') as f:
    Ours_record = pickle.load(f)
Ours_RMSE_P = Ours_record[:,0]
Ours_RMSE_E = Ours_record[:,1]

with open('./supp_summary/LH_GP_record.pkl', 'rb') as f:
    GP_record = pickle.load(f)
GP_RMSE_P = GP_record['LH_GP_P_RMSE']
GP_RMSE_E = GP_record['LH_GP_E_RMSE']

with open('./supp_summary/LH_PR_record.pkl', 'rb') as f:
    PR_record = pickle.load(f)
PR_RMSE_P = PR_record['LH_PR_P_RMSE']
PR_RMSE_E = PR_record['LH_PR_E_RMSE']

Vinet_RMSE_P = [41.494, 25.730, 2.755, 2.150 ]       # from supplement table6
BM_RMSE_P = [1.334, 0.979, 0.959, 1.035 ]

with open('./supp_summary/LH_MG_record.pkl', 'rb') as f:
    MG_record = pickle.load(f)
MG_RMSE_E = MG_record['LH_MG_RMSE']

# %%
'''
dataset
'''
Ours_P = Ours_record[:,2]
GP_P = GP_record['LH_GP_P_RMSE']
PR_P = PR_record['LH_PR_P_RMSE']
BM = [69.699, 38.087, 2.772, 1.722 ]            # from excel 'RMSE_PVTE'
Vinet = [41.494, 25.730, 2.755, 2.150 ]

Ours_E = Ours_record[:,3]
GP_E = GP_record['LH_GP_E_RMSE']
PR_E = PR_record['LH_PR_E_RMSE']
MG = MG_record['LH_MG_RMSE']

fig,ax = plt.subplots(1,3, figsize = (10,4), dpi = 300)
ax[0].add_patch(Rectangle((0,0), 400, 4500,facecolor='b', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 500, 5500,  facecolor='b', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 650, 6500,  facecolor='r', alpha=0.4))
ax[0].add_patch(Rectangle((0,0), 700, 7500,  facecolor='pink', alpha=0.4))
ax[0].scatter(data[:,2], data[:,1])
ax[0].text(20, 200, 'I', fontsize=14)
ax[0].text(420, 200, 'II', fontsize=14)
ax[0].text(520, 200, 'III', fontsize=14)
ax[0].text(650, 200, 'IV', fontsize=14)
ax[0].set_ylabel('Temperature (K)')
ax[0].set_xlabel('Pressure (GPa)')

ax[1].plot(Ours_P,'^--', label='Ours', c='#D60C00',zorder=5,ms = 8,linewidth=2)
ax[1].plot(GP_P,'o-', label=r'PR ($\it{d = 3}$) + GP', c='#3182BD',ms = 9, markerfacecolor='none',markeredgewidth=2)
ax[1].plot(PR_P,'^--', label=r'PR ($\it{d = 3}$)', c='#7B4173')
ax[1].plot(BM,'^--', label='MGD + BM', c='#FD8D3C')
ax[1].plot(Vinet,'^--', label='MGD + Vinet' , c='#FDAE6B')


ax[1].set_xticks(np.arange(4))
ax[1].set_xticklabels(['I', 'II', 'III', 'IV'])
# ax[1].set_yscale('log', base=10)
# ax[1].set_yticks([10,100,300])
ax[1].set_ylabel('RMSE-P (GPa)')
ax[1].legend(fontsize =8)


ax[2].plot(Ours_E,'^--', label='Ours', c='#D60C00', zorder=4, ms = 8, linewidth=2)
ax[2].plot(GP_E,'o-', label=r'PR ($\it{d = 3}$) + GP', c='#3182BD',ms = 9,markerfacecolor='none', markeredgewidth=2)
ax[2].plot(PR_E,'^--', label=r'PR ($\it{d = 3}$)', c='#7B4173')
ax[2].plot(MG,'^--', label='MG', c='#FD8D3C')

ax[2].plot()
ax[2].set_xticks(np.arange(4))
ax[2].set_xticklabels(['I', 'II', 'III', 'IV'])
ax[2].set_yticks([0.25, 0.5, 0.75, 1, 1.25])
ax[2].set_ylabel('RMSE-E (eV/atom)',)
ax[2].legend(loc='upper right',fontsize = 8)
plt.tight_layout()


