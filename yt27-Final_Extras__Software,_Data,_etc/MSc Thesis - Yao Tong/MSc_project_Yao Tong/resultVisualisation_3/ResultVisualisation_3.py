import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import defaultdict
### dataset preparation
# get the path of the given data
path = os.path.dirname(os.path.abspath(__file__)) + '/result3.csv'
result = pd.read_csv(path)
algorithm_list = result['Algorithm']
model_list = result['Model_type']
RMSE_list = result['RMSE']
RMSE = defaultdict(list)
R2_list = result['R2']
R2 = defaultdict(list)
for i,n in enumerate(algorithm_list):
    if n == 0:
        R2['Algorithm0'].append(R2_list[i])
        RMSE['Algorithm0'].append(RMSE_list[i])
    if n == 1:
        R2['Algorithm1'].append(R2_list[i])
        RMSE['Algorithm1'].append(RMSE_list[i])
    if n == 2:
        R2['Algorithm2'].append(R2_list[i])
        RMSE['Algorithm2'].append(RMSE_list[i])
    if n == 3:
        R2['Algorithm3'].append(R2_list[i])
        RMSE['Algorithm3'].append(RMSE_list[i])
    if n == 4:
        R2['Algorithm4'].append(R2_list[i])
        RMSE['Algorithm4'].append(RMSE_list[i])
    if n == 5:
        R2['Algorithm5'].append(R2_list[i])
        RMSE['Algorithm5'].append(RMSE_list[i])
    if n == 6:
        R2['Algorithm6'].append(R2_list[i])
        RMSE['Algorithm6'].append(RMSE_list[i])

plt.title("Comparison of R2 when using different data balance algorithms (Mode type: " + model_list[0]+ ")")
plt.ylabel("R2")
plt.xlabel("window size / movement way")
plt.grid(color='grey', axis = 'y', linestyle=':', linewidth=1)
bins = np.arange(12)
width = 0.4
plt.ylim(0,1.0)
plt.xticks(bins,('5/1','5/2','5/3','5/5','10/1','10/3','10/5','10/10','20/1','20/7', '20/10', '20/20'))
plt.plot(bins,R2['Algorithm0'],label = 'without using algorithm')
plt.plot(bins,R2['Algorithm1'],label = 'use algorithm1')
plt.plot(bins,R2['Algorithm2'],label = 'use algorithm2')
plt.plot(bins,R2['Algorithm3'],label = 'use algorithm3')
plt.plot(bins,R2['Algorithm4'],label = 'use algorithm4')
plt.plot(bins,R2['Algorithm5'],label = 'use algorithm5')
plt.plot(bins,R2['Algorithm6'],label = 'use algorithm6')

plt.legend()
plt.show()

plt.title("Comparison of RMSE when using different data balance algorithms (Mode type: " + model_list[0]+ ")")
plt.ylabel("RMSE")
plt.xlabel("window size / movement way")
plt.grid(color='grey', axis = 'y', linestyle=':', linewidth=1)
bins = np.arange(12)
width = 0.4
#plt.ylim(0,1.0)
plt.xticks(bins,('5/1','5/2','5/3','5/5','10/1','10/3','10/5','10/10','20/1','20/7', '20/10', '20/20'))
plt.plot(bins,RMSE['Algorithm0'],label = 'without using algorithm')
plt.plot(bins,RMSE['Algorithm1'],label = 'use algorithm1')
plt.plot(bins,RMSE['Algorithm2'],label = 'use algorithm2')
plt.plot(bins,RMSE['Algorithm3'],label = 'use algorithm3')
plt.plot(bins,RMSE['Algorithm4'],label = 'use algorithm4')
plt.plot(bins,RMSE['Algorithm5'],label = 'use algorithm5')
plt.plot(bins,RMSE['Algorithm6'],label = 'use algorithm6')

plt.legend()
plt.show()

result2 = pd.read_csv(path, usecols = ['Model_type','Algorithm','Dataset_size','RMSE','R2'])
result2 = result2.sort_values(by = ['Algorithm','Dataset_size'])
result2 = result2.reset_index(drop=True)
Dataset_size_list = result2['Dataset_size']
Dataset_size = defaultdict(list)
algorithm_list2 = result2['Algorithm']
model_list2 = result2['Model_type']
RMSE_list2 = result2['RMSE']
RMSE_2 = defaultdict(list)
R2_list2 = result2['R2']
R2_2 = defaultdict(list)
for i,n in enumerate(algorithm_list2):
    if n == 0:
        R2_2['algorithm0'].append(R2_list2[i])
        RMSE_2['algorithm0'].append(RMSE_list2[i])
        Dataset_size['algorithm0'].append(Dataset_size_list[i])
    if n == 1:
        R2_2['algorithm1'].append(R2_list2[i])
        RMSE_2['algorithm1'].append(RMSE_list2[i])
        Dataset_size['algorithm1'].append(Dataset_size_list[i])
    if n == 2:
        R2_2['algorithm2'].append(R2_list2[i])
        RMSE_2['algorithm2'].append(RMSE_list2[i])
        Dataset_size['algorithm2'].append(Dataset_size_list[i])
    if n == 3:
        R2_2['algorithm3'].append(R2_list2[i])
        RMSE_2['algorithm3'].append(RMSE_list2[i])
        Dataset_size['algorithm3'].append(Dataset_size_list[i])
    if n == 4:
        R2_2['algorithm4'].append(R2_list2[i])
        RMSE_2['algorithm4'].append(RMSE_list2[i])
        Dataset_size['algorithm4'].append(Dataset_size_list[i])
    if n == 5:
        R2['Algorithm5'].append(R2_list[i])
        RMSE['Algorithm5'].append(RMSE_list[i])
        Dataset_size['algorithm5'].append(Dataset_size_list[i])
    if n == 6:
        R2['Algorithm6'].append(R2_list[i])
        RMSE['Algorithm6'].append(RMSE_list[i])
        Dataset_size['algorithm6'].append(Dataset_size_list[i])

for k in  Dataset_size.keys():
    Title = "The change of R2 when using the data balance " + k + " (Mode type: " + model_list2[0]+ ")"
    plt.title(Title)
    plt.ylabel("R2")
    plt.xlabel("Dataset size")
    plt.ylim(0,1.0)
    plt.grid(color='grey', linestyle=':', linewidth=1)
    bins2 = np.arange(len(Dataset_size[k]))
    plt.xticks(bins2,Dataset_size[k])
    plt.plot(bins2,R2_2[k],label = k)
    plt.legend()
    plt.show()
