import numpy as np
import scipy.sparse as ss
import pandas as pd
adj_data=pd.read_csv("F:\\图神经网络时间序列预测\\异常检测修改\\SMAP_adj.csv").values
a=np.zeros((adj_data.shape[0],adj_data.shape[1]))
for i in range(adj_data.shape[0]):
    for j in range(adj_data.shape[1]):
        a[i][j]=adj_data[i][j]
        if i==j:
            a[i][j]=1


matrix=ss.csc_matrix(a)
print(matrix)
ss.save_npz("/data/smap/adj.npz",matrix,compressed=True)
