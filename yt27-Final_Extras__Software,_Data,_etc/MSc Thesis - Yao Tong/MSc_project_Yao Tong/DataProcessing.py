import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import random
import functools
import operator
from os import listdir
from collections import defaultdict
from random import seed
from math import log,log1p

class DataProcessing:

    def __init__(self,parent_folder,sliding_window,move):
        self.parent_folder = parent_folder
        self.sliding_window = sliding_window
        self.move = move
        self.file_name_list = []
        self.file_name_list = self.fileNameList()
        self.input_file_list = []
        self.input_file_list = self.inputFileList()
        self.output_file_list = []
        self.output_file_list = self.outputFileList()
        self.condition = {}
        self.condition = self.setCondition()
        self.input_dataset = []
        self.output_dataset = []
        self.new_input_dataset = []
        self.new_output_dataset = []
        self.distribution_dictionary1 = defaultdict(list)
        self.distribution_dictionary2= defaultdict(list)
        self.new_distribution_dictionary1 = defaultdict(list)
        self.new_distribution_dictionary2 = defaultdict(list)

    def fileNameList(self): # file_name_list collects all input and output file names
        self.file_name_list = os.listdir(self.parent_folder)
        return self.file_name_list

    def inputFileList(self): # input_file_list collects all input file names
        for f in self.file_name_list:
            if f[0] is "X":
                self.input_file_list.append(f)
        return self.input_file_list

    def outputFileList(self): # output_file_list collects all output file names
        for f in self.input_file_list:
            f = "Y" + f[1:]
            self.output_file_list.append(f)
        return self.output_file_list

    def calcuInputAverage(self): # average calculation procedure to get the new inputs of the original data set
        for f in self.input_file_list:
            path = self.parent_folder + f
            data = np.load(path, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
            samples = len(data)
            features = len(data[0])
            counter = self.sliding_window
            while counter <= samples:
                temp = []
                for j in range(0,features):
                    sum = 0.0
                    average = 0.0
                    for i in range(counter-self.sliding_window,counter):
                        sum = sum + data[i][j]
                    average = sum/self.sliding_window
                    temp.append(average)
                self.input_dataset.append(temp)
                counter += self.move

    def calcuOutputAverage(self): # average calculation procedure to get the new outputs of the original data set
        for f in self.output_file_list:
            path = self.parent_folder + f
            data = np.load(path, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')
            samples = len(data)
            counter = self.sliding_window
            while counter <= samples:
                sum = 0.0
                average = 0.0
                for i in range(counter-self.sliding_window,counter):
                    sum = sum + data[i]
                average = sum/self.sliding_window
                self.output_dataset.append(average)
                counter += self.move

    def calcuOutputValueDistribution1(self): # calculate the distribution of outputs
        for i,n in enumerate(self.output_dataset):
            for k,v in self.condition.items():
                if v[1] != 1.0:
                    if n>=v[0] and n<v[1]:
                        self.distribution_dictionary1[k].append(i) # the elements of the list under the key are
                else:                                              # the collection of the indexes of the qualified observations
                    if n>=v[0] and n<=v[1]:
                        self.distribution_dictionary1[k].append(i)

    def transformOutput(self):
        for i,n in enumerate(self.new_output_dataset):
            if n == 0:
                n = 0.00000000001
            if n == 1:
                n = 0.99999999999
            self.new_output_dataset[i] = log(n/(1-n))

    def calcuOutputValueDistribution2(self):
        for k,v in self.condition.items():
            self.distribution_dictionary2[k] = len(self.distribution_dictionary1[k]) # store the numbers of the observations under the key
        # pd.DataFrame.from_dict(data = self.distribution_dictionary2,orient='index').to_csv('distribution dictionary2.csv', header=False)
        # print("success3")

    def preserveAlgorithm0(self):
        self.calcuOutputValueDistribution2()
        for (k,v) in  self.distribution_dictionary2.items():
            self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]
            self.new_distribution_dictionary2[k] = self.distribution_dictionary2[k]

    def preserveAlgorithm1(self): # Data balance algorithm1
        seed(1)
        self.calcuOutputValueDistribution2()
        d = sorted(self.distribution_dictionary2.values())
        m = 0 # try to assign the minimum(excluding 0) to m
        for n in d:
            if n > 0:
                m = n
                break
        for (k,v) in  self.distribution_dictionary2.items():
            if v > m:
                self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], m)
                self.new_distribution_dictionary2[k] = m
            else:
                self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]
                self.new_distribution_dictionary2[k] = v

    def preserveAlgorithm2(self): # Data balance algorithm2
        seed(1)
        self.calcuOutputValueDistribution2()
        d = sorted(self.distribution_dictionary2.values())
        l = len(d)
        if l % 2 == 0:   # to get median
            median = (d[l//2]+d[l//2-1])//2
        if l % 2 == 1:
            median = d[(l-1)//2]
        for (k,v) in  self.distribution_dictionary2.items():
            if v > median:
                self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], median)
                self.new_distribution_dictionary2[k] = median
            else:
                self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]
                self.new_distribution_dictionary2[k] = v

    def preserveAlgorithm3(self): # Data balance algorithm3
        seed(1)
        self.calcuOutputValueDistribution2()
        d = sorted(self.distribution_dictionary2.values(), reverse = True)
        second_max = d[1] # second_max is the second maximum value in new_distribution_dictionary2
        for (k,v) in  self.distribution_dictionary2.items():
            if v > second_max:
                self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], second_max)
                self.new_distribution_dictionary2[k] = second_max
            else:
                self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]
                self.new_distribution_dictionary2[k] = v

    def preserveAlgorithm4(self): # Data balance algorithm4()
        seed(1)
        self.calcuOutputValueDistribution2()
        d = list(self.distribution_dictionary2.values())
        index = 0
        l = len(d)-1
        for (k,v) in  self.distribution_dictionary2.items():
            index_ = l - index
            number = (min(d[index],d[index_]))
            self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], number)
            self.new_distribution_dictionary2[k] = number
            index = index + 1

    def preserveAlgorithm5(self): # Data balance algorithm4()
        seed(1)
        self.calcuOutputValueDistribution2()
        d = sorted(self.distribution_dictionary2.values(), reverse = True)
        second_max = d[1] # second_max is the second maximum value in new_distribution_dictionary2
        for (k,v) in  self.distribution_dictionary2.items():
            if v > second_max:
                self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], second_max)
            else:
                self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]+[random.choice(self.distribution_dictionary1[k]) for _ in range(second_max-v)]
            self.new_distribution_dictionary2[k] = second_max

    def preserveAlgorithm6(self): # Data balance algorithm2
        seed(1)
        self.calcuOutputValueDistribution2()
        d = sorted(self.distribution_dictionary2.values())
        l = len(d)
        if l % 2 == 0:   # to get median
            median = (d[l//2]+d[l//2-1])//2
        if l % 2 == 1:
            median = d[(l-1)//2]
        for (k,v) in  self.distribution_dictionary2.items():
            if v > median:
                self.new_distribution_dictionary1[k] = random.sample(self.distribution_dictionary1[k], median)
            else:
                self.new_distribution_dictionary1[k] = self.distribution_dictionary1[k]+[random.choice(self.distribution_dictionary1[k]) for _ in range(median-v)]
            self.new_distribution_dictionary2[k] = median


    def cleanDataset(self): # update the dataset
        seed(2)
        index = self.new_distribution_dictionary1.values()
        index = functools.reduce(operator.add,index )
        random.shuffle(index)
        temp1 = []
        temp2 = []
        for i in index:
            temp1.append(self.output_dataset[i])
            temp2.append(self.input_dataset[i])
        self.new_output_dataset = temp1
        self.new_input_dataset = temp2

    def setCondition(self):
        if self.sliding_window > 0 and self.sliding_window < 10:
            self.condition= {'[0.0,0.2)':[0.0,0.2],
                             '[0.2,0.4)':[0.2,0.4],
                             '[0.4,0.6)':[0.4,0.6],
                             '[0.6,0.8)':[0.6,0.8],
                             '[0.8,1.0]':[0.8,1.0]}

        if self.sliding_window >= 10:
            self.condition =  {'[0.0,0.1)':[0.0,0.1],
                               '[0.1,0.2)':[0.1,0.2],
                               '[0.2,0.3)':[0.2,0.3],
                               '[0.3,0.4)':[0.3,0.4],
                               '[0.4,0.5)':[0.4,0.5],
                               '[0.5,0.6)':[0.5,0.6],
                               '[0.6,0.7)':[0.6,0.7],
                               '[0.7,0.8)':[0.7,0.8],
                               '[0.8,0.9)':[0.8,0.9],
                               '[0.9,1.0]':[0.9,1.0]}
        return self.condition
