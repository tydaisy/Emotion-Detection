import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import matplotlib
from DataVisualisation import DataVisualisation
from DataProcessing import DataProcessing
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import metrics
from collections import defaultdict
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from math import exp,expm1
from sklearn.feature_selection import SelectKBest, f_regression
from statistics import mean

def distribution(data):#
    data_distribution = [0] * len(dataset.condition)
    scaler = len(dataset.condition)
    for n in data:
        for v in dataset.condition.values():
            if v[1] != 1.0:
                if n>=v[0] and n<v[1]:
                    data_distribution[int(v[0]*scaler)] += 1
            else:
                if n>=v[0] and n<=v[1]:
                    data_distribution[int(v[0]*scaler)] += 1
    return (data_distribution)

def transbackOuput(data):
    output = []
    for n in data:
        n = exp(n)/(1+exp(n))
        output.append(n)
    return output

### dataset preparation
# get the path of the given data
parent_folder = os.path.dirname(os.path.abspath(__file__)) + '/data/'
# set 1st, 2nd parameters: 'window size' and 'window movement'
sliding_window_size = int(input("Please input the size of the sliding window: "))
window_movement = int(input("Please input the movement way of the sliding window: "))
dataset = DataProcessing(parent_folder,sliding_window_size,window_movement)
# do average to get the original data
dataset.calcuInputAverage()
dataset.calcuOutputAverage()
dataset.calcuOutputValueDistribution1()
# data balance to get the updated dataset
str1 = input("\nWhich algorithm do you want to use to balance the dataset?\nA. Algorithm0\nB. Algorithm1 \nC. Algorithm2\nD. Algorithm3\nE. Algorithm4\nF. Algorithm5\nG. Algorithm6\n" )
if str1 == 'A':
    dataset.preserveAlgorithm0()
    algorithm = 'Algorithm0'
if str1 == 'B':
    dataset.preserveAlgorithm1()
    algorithm = 'Algorithm1'
if str1 == 'C':
    dataset.preserveAlgorithm2()
    algorithm = 'Algorithm2'
if str1 == 'D':
    dataset.preserveAlgorithm3()
    algorithm = 'Algorithm3'
if str1 == 'E':
    dataset.preserveAlgorithm4()
    algorithm = 'Algorithm4'
if str1 == 'F':
    dataset.preserveAlgorithm5()
    algorithm = 'Algorithm5'
if str1 == 'G':
    dataset.preserveAlgorithm6()
    algorithm = 'Algorithm6'
dataset.cleanDataset()

### visualise output
draw1 = DataVisualisation()
str2 = input("\nWould you like to visualise the original output data?\nPress 1 for yes, and press any other keys for no:")
if str2 == '1':
    draw1.drawOutput(dataset.output_dataset,"The original outputs visualisation")
str2 = input("\nWould you like to visualise the distribution of original output data?\nPress 1 for yes, and press any other keys for no:")
if str2 == '1':
    draw1.outputDistribution(dataset.condition,dataset.distribution_dictionary2.values(),"The distribution of the original outputs")
str2 = input("\nWould you like to visualise the updated output data?\nPress 1 for yes, and press any other keys for no:")
if str2 == '1':
    draw1.drawOutput(dataset.new_output_dataset,"The updated outputs visualisation")
str2 = input("\nWould you like to visualise the distribution of updated output data?\nPress 1 for yes, and press any other keys for no:")
if str2 == '1':
    draw1.outputDistribution(dataset.condition,dataset.new_distribution_dictionary2.values(),"The distribution of the updated outputs")
str2 = input("\nWould you like to visualise the comparison of the distribution between original output data and updated output data?\nPress 1 for yes, and press any other keys for no:")
if str2 == '1':
    draw1.compareDistribution(dataset.condition,dataset.distribution_dictionary2.values(),dataset.new_distribution_dictionary2.values(),'original output','updated output')

### select model type: LR, SVR, LOR
str3 = input("\nWhat's model do you want to use?\nA. Linear Regression Model\nB. Support Vector Regression Model\nC. Logistic Regression Model\n" )
model_type = []
if str3 == 'A':
    model = linear_model.LinearRegression()
    model_type = 'LR'
if str3 == 'B':
    model = SVR(kernel="poly")
    model_type = 'SVR'
if str3 == 'C':
    dataset.transformOutput()
    model = linear_model.LinearRegression()
    model_type = 'LOR'

### show the distribution of each feature in one page
pd.DataFrame(dataset.new_input_dataset).hist()
plt.title('')
plt.show('r')

X = np.array(dataset.new_input_dataset)
y = np.array(dataset.new_output_dataset)

### data standarsation
X = preprocessing.StandardScaler().fit_transform(X)
percent = 0.1
r = int(len(X)*percent)
X_test = X[:r]
y_test = y[:r]
X_train_validation = X[r:]
y_train_validation = y[r:]

### k-fold validation
# set "k" value of k-fold validation
k = 10
kf = KFold(n_splits=k)
# train model with 10-fold validation
y_predict = []
# all_RMSE = all_R2 = all_accuracy = 0.0
feature_scores_RMSE = [] # average score collection based on different num of features for validation dataset
feature_scores_R2 = []
feature_scores_RMSE2 = [] # average score collection based on different num of features for test dataset
feature_scores_R22 = []
all_alRMSE = []
all_alR2 = []
all_mask = []
all_alIntercepts = []
all_alCoefficients = []
all_alVectors = []
all_alRMSE2 = []
all_alR22 = []
for i in range(1,43):
    all_RMSE = []
    all_R2 = []
    mask = []
    all_Intercepts = []
    all_Coefficients = []
    all_vectors = []
    all_RMSE2 = []
    all_R22 = []
    for train_index, test_index in kf.split(X_train_validation):
        y_true = []
        y_true = y_train_validation[test_index]
        y_true2 = []
        y_true2 = y_test
        # feature selection
        select = SelectKBest(score_func=f_regression, k=i)
        select.fit(X_train_validation[train_index],y_train_validation[train_index])
        mask = select.get_support()
        x_train_selected = select.transform(X_train_validation[train_index])
        x_validation_selected = select.transform(X_train_validation[test_index])
        x_test_selected = select.transform(X_test)
        model.fit(x_train_selected,y_train_validation[train_index])
        y_predict = model.predict(x_validation_selected)
        y_predict2 = model.predict(x_test_selected)
        RMSE = R2 = 0.0
        RMSE2 = R22 = 0.0
        RMSE = np.sqrt(metrics.mean_squared_error(y_true,y_predict))
        R2 = r2_score(y_true,y_predict)
        RMSE2 = np.sqrt(metrics.mean_squared_error(y_true2,y_predict2))
        R22 = r2_score(y_true2,y_predict2)
        all_RMSE.append(RMSE)
        all_R2.append(R2)
        all_RMSE2.append(RMSE2)
        all_R22.append(R22)

        if str3 == 'A' or str3 == 'C':
            intercept  = model.intercept_
            coefficients = model.coef_
            all_Intercepts.append(intercept)
            all_Coefficients.append(coefficients)

        if str3 == 'B':
            vector_num = len(model.support_)
            all_vectors.append(vector_num)
    all_mask.append(mask)
    average_RMSE = mean(all_RMSE)
    average_R2 = mean(all_R2)
    average_RMSE2 = mean(all_RMSE2)
    average_R22 = mean(all_R22)

    feature_scores_RMSE.append(average_RMSE)
    feature_scores_R2.append(average_R2)
    feature_scores_RMSE2.append(average_RMSE2)
    feature_scores_R22.append(average_R22)

    all_alRMSE.append(all_RMSE)
    all_alR2.append(all_R2)
    all_alRMSE2.append(all_RMSE2)
    all_alR22.append(all_R22)
    all_alIntercepts.append(all_Intercepts)
    all_alCoefficients.append(all_Coefficients)
    all_alVectors.append(all_vectors)
if model_type == 'LOR':
    draw1.featureScore(feature_scores_RMSE,'RMSE for validation dataset',feature_scores_RMSE2,'RMSE for test dataset',model_type)
    draw1.featureScore(feature_scores_R2,'R2 for validation dataset',feature_scores_R22,'R2 for test dataset',model_type)
else:
    # draw1.featureScore(feature_scores_RMSE,'RMSE',feature_scores_R2,'R2',model_type)
    draw1.featureScore2(feature_scores_RMSE,'RMSE for validation dataset',feature_scores_R2,'R2 for validation dataset',feature_scores_RMSE2,'RMSE for test dataset',feature_scores_R22,'R2 for test dataset',model_type)

### draw mask
optimal = feature_scores_R2.index(max(feature_scores_R2))
plt.matshow(all_mask[optimal].reshape(1,-1),cmap = 'gray_r')
plt.xlabel("Sample index",fontsize = 12)
plt.ylabel('')
plt.show()
print(all_mask[optimal])
### the change of RMSE and R2 in optimal features when using 10-fold cross validation
draw1.compareScore2(mean(all_alR2[optimal]),all_alR2[optimal],mean(all_alR22[optimal]),all_alR22[optimal],'R2')
draw1.compareScore2(mean(all_alRMSE[optimal]),all_alRMSE[optimal],mean(all_alRMSE2[optimal]),all_alRMSE2[optimal],'RMSE')

### show information table
model_name = []
criteria = defaultdict(list)
parameters = defaultdict(list)
if str3 == 'A' or str3 == 'C':
    parameters["Intercept"] = all_alIntercepts[optimal]
    parameters["Coefficients"] = all_alCoefficients[optimal]
if str3 == 'B':
    parameters["# of SVs"] = all_alVectors[optimal]
criteria["RMSE"] = all_alRMSE[optimal]
criteria["R2"] = all_alR2[optimal]
for n in range(0,k):
    model_name.append(model_type)
times = np.arange(1,k+1)
draw1.autotable(times,model_name,parameters,criteria)
print("The algorithm used in balancing the dataset: ", algorithm)
print("The size of the dataset: ", len(y_train_validation))
print("# of features: ", feature_scores_R2.index(max(feature_scores_R2))+1)
print ("mean RMSE:", mean(all_alRMSE[optimal]))
print ("mean R2:", mean(all_alR2[optimal]))
print ("\nmean RMSE2:", mean(all_alRMSE2[optimal]))
print ("mean R22:", mean(all_alR22[optimal]))

### visualise y_pred VS y_true
str4 = input("\nWould you like to see y_pred VS y_true? \nPress 1 for yes, and press any other keys for no:")
if str4 =='1':
    for train_index, test_index in kf.split(X_train_validation):
        y_true2 = []
        y_true2 = y_test
        select = SelectKBest(score_func=f_regression, k=optimal+1)
        select.fit(X_train_validation[train_index],y_train_validation[train_index])
        x_train_selected = select.transform(X_train_validation[train_index])
        x_test_selected = select.transform(X_test)
        model.fit(x_train_selected,y_train_validation[train_index])
        y_predict2 = model.predict(x_test_selected)
        if str3 == 'C':
            y_predict2 = transbackOuput(y_predict2)
            y_true2 = transbackOuput(y_true2)
        draw1.y_pred_VS_y_true_2D(y_true2,y_predict2)
        draw1.y_pred_VS_y_true(y_true2,y_predict2)
