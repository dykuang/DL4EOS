#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

# data

with open('../summary/CV_PVTE_record_CurrentBest.pkl', 'rb') as f:
    Ours_record = pickle.load(f)
Ours_RMSE_P = Ours_record[:,0]
Ours_RMSE_E = Ours_record[:,1]

with open('./supp_summary/CV_GP_record.pkl', 'rb') as f:
    GP_record = pickle.load(f)
GP_RMSE_P = GP_record['CV_GP_P_RMSE_d3']
GP_RMSE_E = GP_record['CV_GP_E_RMSE_d3']

with open('./supp_summary/CV_PR_record.pkl', 'rb') as f:
    PR_record = pickle.load(f)
PR_RMSE_P = PR_record['CV_PR_P_RMSE_d3']
PR_RMSE_E = PR_record['CV_PR_E_RMSE_d3']

Vinet_RMSE_P = [0.798, 39.394, 2.727, 2.757, 3.256 ]
BM_RMSE_P = [0.958, 21.076, 2.498, 3.224, 3.409 ]

with open('./supp_summary/CV_MG_record.pkl', 'rb') as f:
    MG_record = pickle.load(f)
MG_RMSE_E = MG_record['CV_MG_RMSE']


#%%

data_P = {
    'Folds': ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'],
    'Ours': Ours_RMSE_P,
    'PR (d = 3) + GP': GP_RMSE_P,
    'PR (d = 3)': PR_RMSE_P,
    'Vinet': Vinet_RMSE_P,
    'BM': BM_RMSE_P,
}
data_E = {
    'Folds': ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'],
    'Ours': Ours_RMSE_E,
    'PR (d = 3) + GP': GP_RMSE_E,
    'PR (d = 3)': PR_RMSE_E,
    'MG': MG_RMSE_E,
}

df_P = pd.DataFrame(data_P)
df_E = pd.DataFrame(data_E)

# plot 
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 12), dpi=150)

# RMSE-P
sns.lineplot(x='Folds', y='Ours', data=df_P, marker='^', label='Ours', ms=12, linewidth=2.3, c='#D60C00',linestyle='--', ax=ax1,zorder=5)
sns.lineplot(x='Folds', y='PR (d = 3) + GP', data=df_P, marker='o', ms=11, label=r'PR ($\it{d = 3}$) + GP', 
             linewidth=2, c='#3182BD', linestyle='-',markerfacecolor='none', markeredgecolor='#3182BD',markeredgewidth=2, ax=ax1)
sns.lineplot(x='Folds', y='PR (d = 3)', data=df_P, marker='^', ms=10, label=r'PR ($\it{d = 3}$)', linewidth=2, c='#7B4173', linestyle='--', ax=ax1)
sns.lineplot(x='Folds', y='Vinet', data=df_P, marker='^', ms=10, label='MGD + Vinet', linewidth=2, c='#FD8D3C', linestyle='--', ax=ax1)
sns.lineplot(x='Folds', y='BM', data=df_P, marker='^', ms=10, label='MGD + BM', linewidth=2, c='#FDAE6B', linestyle='--', ax=ax1)


ax1.set_xlabel('')
ax1.set_ylabel('RMSE-P (GPa)', fontsize=20)
ax1.set_xticks([])
ax1.set_yticks([10, 20, 30, 40, 50])
ax1.tick_params(axis='y', labelsize=20)
ax1.legend( fontsize=17.5,  loc='upper right')

# RMSE-E
sns.lineplot(x='Folds', y='Ours', data=df_E, marker='^', label='Ours', ms=12, linewidth=2.3, c='#D60C00',linestyle='--', ax=ax2)
sns.lineplot(x='Folds', y='PR (d = 3) + GP', data=df_E, marker='o', ms=11, label=r'PR ($\it{d = 3}$) + GP', linewidth=2, c='#3182BD', linestyle='-',markerfacecolor='none', markeredgecolor='#3182BD',markeredgewidth=2, ax=ax2)
sns.lineplot(x='Folds', y='PR (d = 3)', data=df_E, marker='^', ms=10, label=r'PR ($\it{d = 3}$)', linewidth=2, c='#7B4173', linestyle='--', ax=ax2)
sns.lineplot(x='Folds', y='MG', data=df_E, marker='^', ms=10, label='MG', linewidth=2, c='#FD8D3C', linestyle='--', ax=ax2)

ax2.set_xlabel('')
ax2.set_ylabel('RMSE-E (eV/atom)', fontsize=20)
ax2.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
ax2.legend(fontsize=17.5, loc='upper right')

plt.tight_layout()

# %%
