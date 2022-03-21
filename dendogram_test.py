import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist
from dataframe_manipulating import slice_dataset

np.set_printoptions(precision=5, suppress=True)

data = pd.read_csv('set-a.csv', header=0, delimiter=r",")

data = slice_dataset(data, first_cut_threshold=0.7, dropna_first=True, z_scale=True, columns=['Age', 'ICUType', 'ALP_mean', 'ALT_mean', 'AST_mean', 'Albumin_mean', 'BUN_mean', 'Bilirubin_mean', 'Creatinine_mean', 'DiasABP_mean', 'GCS_mean', 'Glucose_mean', 'HCO3_mean', 'HCT_mean', 'HR_mean', 'Lactate_mean', 'MAP_mean', 'Mg_mean', 'NIDiasABP_mean', 'NIMAP_mean', 'NISysABP_mean', 'Na_mean', 'PaCO2_mean', 'PaO2_mean', 'SysABP_mean', 'Urine_mean', 'pH_mean', 'outcome'], do_clustering=False)

Z = linkage(data[data['outcome'] == 1].iloc[:,0:-1], 'ward', metric='euclidean')
c, coph_dists = cophenet(Z, pdist(data[data['outcome'] == 1].iloc[:,0:-1]))
print(c)

plt.figure(figsize=(25,10))
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.show()
