import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def calcuOutputDistribution(data):
    distribution = [0,0,0,0,0,0,0]
    percentage = [0,0,0,0,0,0,0]
    for n in data:
        if n>=0 and n<0.15:
            distribution[0] +=1
        if n>=0.15 and n<0.30:
            distribution[1] +=1
        if n>=0.3 and n<0.45:
            distribution[2] +=1
        if n>=0.45 and n<0.6:
            distribution[3] +=1
        if n>=0.6 and n<0.75:
            distribution[4] +=1
        if n>=0.75 and n<0.9:
            distribution[5] +=1
        if n>=0.9 and n<=1.0:
            distribution[6] +=1
    length = distribution[0] + distribution[1] + distribution[2] + distribution[3] + distribution[4] + distribution[5] + distribution[6]
    for i in range(0,7):
        percentage[i] = distribution[i]/length
    return percentage

### dataset preparation
# get the path of the given data
path = os.path.dirname(os.path.abspath(__file__)) + '/results.csv'
result = pd.read_csv(path)

LR_R2 = calcuOutputDistribution(result['R22'][:84])
LOR_R2 = calcuOutputDistribution(result['R22'][85:169])
SVR_R2 = calcuOutputDistribution(result['R22'][170:237])


plt.title("Comparison distribution of R2 based on different modles (for test dataset)")
plt.ylabel("Percentage")
plt.xlabel("R2")
plt.grid(color='grey', axis = 'y', linestyle=':', linewidth=1)
bins = np.arange(7)
width = 0.2
plt.xticks(bins,('0-0.15','0.15-0.30','0.30-0.45','0.45-0.60','0.60-0.75','0.75-0.90','0.90-1.0'))
plt.bar(bins-width,LR_R2,width = width,color = 'steelblue', label = 'LR')
plt.bar(bins,SVR_R2,width = width,color = 'pink', label = 'SVR')
plt.bar(bins+width,LOR_R2,width = width,color = 'sandybrown', label = 'LOR')
plt.legend()
plt.show()
