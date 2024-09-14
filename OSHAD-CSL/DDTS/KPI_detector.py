from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ads_evt as spot


import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
import math
import numpy as np
plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size':8})
plt.figure(figsize=(10, 12))
plt.tight_layout()
plt.subplots_adjust(left=0.1,bottom=None,right=None,top=None,wspace=None,hspace=None)

data_predict=pd.read_csv("./data/KPI_predictvalue.csv")
data_real=pd.read_csv("./data/KPI_real.csv")
error=(data_predict-data_real).abs()

label=pd.read_csv("./data/label.csv")[61096-19855-2+1000:].reset_index()
system_label=pd.read_csv("./data/system_label.csv")['label'][61096-19855-2+1000:].reset_index()
init_result=pd.read_csv("./data/initial_result.csv")
columns=data_predict.columns
window_size=100
init_data = 1000
proba = 1e-3
depth = 450

start_time = time.time()

for i in range(len(columns)):
    print(i)
    nor_value=[]
    for j in range(len(data_real)-1):
        mean_value=np.mean(error[columns[i]].iloc[j:j+window_size])
        std_value=np.std(error[columns[i]].iloc[j:j+window_size])
        nor_value.append((error[columns[i]].iloc[j]-mean_value)/std_value)
    data=np.array(error[columns[i]])
    models: List[spot.SPOTBase] = [
         spot.dSPOT(q=proba, depth=depth),
    ]
    for alg in models:
        alg.fit(init_data=init_data, data=data)
        alg.initialize()
        results = alg.run()
        thresholds=results['thresholds']


    m_label=[]

    for j in range(1000,len(data_real)):
        if error[columns[i]].iloc[j]>thresholds[j-1000]:
            m_label.append(1)
        else:
            m_label.append(0)
    init_result[columns[i]]=pd.DataFrame(m_label)



# judge=pd.read_csv("F:\\图神经网络时间序列预测\\anomaly detetction\\judge4.csv")
# for i in range(len(judge)):
#     node=judge['parameter'].iloc[i]
#     cause_array=judge['cause'].iloc[i].split(";")
#     print(i)
#     for j in range(len(init_result)):
#         if init_result[node].iloc[j] == 0:
#             cause_count=0
#             for k in range(len(cause_array)):
#                 if init_result[cause_array[k]].iloc[j] == 1:
#                     cause_count=cause_count+1
#             if (cause_count/len(cause_array))>0.9:
#                 init_result[node].iloc[j] = 1
#         if init_result[node].iloc[j] == 1:
#             cause_count=0
#             for k in range(len(cause_array)):
#                 if init_result[cause_array[k]].iloc[j] == 0:
#                     cause_count=cause_count+1
#             if (cause_count/len(cause_array))>0.9:
#                 init_result[node].iloc[j] = 0

end_time = time.time()  # 程序结束后再次获取当前时间

elapsed_time = end_time - start_time  # 计算程序运行时间

print(f"running time：{elapsed_time}s")

init_result.to_csv("./data/final_result.csv",index=False)
array_recall=[]
array_precision=[]
array_fscore=[]
TP_total = 0
FP_total = 0
FN_total = 0
TN_total = 0
for i in range(len(columns)):
    print(i)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for j in range(len(init_result)):
        if init_result[columns[i]].iloc[j] == 1 and label[columns[i]].iloc[j] == 1:
            TP = TP + 1
            TP_total=TP_total+1
        elif init_result[columns[i]].iloc[j] == 1 and label[columns[i]].iloc[j] == 0:
            FP = FP + 1
            FP_total=FP_total+1
        elif init_result[columns[i]].iloc[j] == 0 and label[columns[i]].iloc[j] == 1:
            FN = FN + 1
            FN_total=FN_total+1
        else:
            TN = TN + 1
            TN_total=TN_total+1

marked_system_label=init_result.sum(axis=1)

system_TP=0
system_FP=0
system_FN=0
system_TN=0
for i in range(len(marked_system_label)):
    if marked_system_label[i]>0 and system_label['label'].iloc[i]==1:
        system_TP=system_TP+1
    elif marked_system_label[i]>0 and system_label['label'].iloc[i]==0:
        system_FP=system_FP+1
    elif marked_system_label[i]==0 and system_label['label'].iloc[i]==1:
        system_FN=system_FN+1
    else:
        system_TN=system_TN+1
recall_system = system_TP / (system_TP + system_FN)
precision_system = system_TP / (system_TP + system_FP)
fpr=system_FP/(system_FP+system_TN)

print("precision")
print(precision_system)
print("recall")
print(recall_system)
print("F1-score")
print(2 * recall_system * precision_system / (recall_system + precision_system))
print("fpr")
print(fpr)


