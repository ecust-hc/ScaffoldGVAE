import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Lipinski, MACCSkeys
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import warnings
import pandas as pd
from rdkit.Chem import AllChem
warnings.filterwarnings('ignore')

df_all = pd.read_csv('D:\PythonProject\Project_VAE\data\LRRK2\\res_lrrk2_sample.csv')
df = df_all[87315:]
df['mol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
#print(df['mol'])
df['ECFP'] = df['SMILES'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x),3,2048))
print("fingerprint done")
X = np.array(list(df['ECFP'].values))
X_tsne = TSNE(random_state=20221025, n_components=2).fit_transform(X)
print("t-SNE done")
df['component1'] = X_tsne[:, 0]
df['component2'] = X_tsne[:, 1]

# Analysis chemical space
fig = plt.figure(figsize=(8, 8))
ax = plt.subplot(aspect='equal')

df_293T = df[df['LABELS']==1]
df_mv411 = df[df['LABELS']==2]
# df_overlap = df[df[mode+'_label']==3]
plt.title('(D)', fontdict={'family': 'Times New Roman', 'size': 16}, x=-0.1, y=1.0, weight='bold')
# ax.scatter(df['component1'], df['component2'], s=20)
ax.scatter(df_293T['component1'], df_293T['component2'], s=20, c='c',label='Gen')
ax.scatter(df_mv411['component1'], df_mv411['component2'], s=20,c = 'r',label = 'Act')
# ax.scatter(df_overlap['component1'], df_overlap['component2'], s=100, marker='*', c='r')
ax.axis('on')
ax.axis('tight')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(['All Mol','_293T','_mv411'], fontsize=15)
ax.legend(['Gen', 'Act'], fontsize=15)
plt.savefig('t-SNE_lrrk.png', dpi=300)
plt.show()