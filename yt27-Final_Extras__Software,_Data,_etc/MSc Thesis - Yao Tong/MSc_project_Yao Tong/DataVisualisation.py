import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from mpl_toolkits.mplot3d import axes3d

class DataVisualisation:
    def __init__(self):
        pass

    def drawOutput(self,data,label1): # display the output value of the index of the corresponding observation
        length = len(data)
        x = np.arange(length)
        y = data
        new_ticks = np.linspace(0,1,11)
        plt.title(label1)
        plt.xlabel("Index")
        plt.ylabel("Output")
        plt.xlim(0,100)
        plt.ylim(-0.03,1.03)
        plt.yticks(new_ticks)
        plt.grid(color = "grey", linestyle = ":", linewidth = 0.9)
        if label1 == "The original outputs visualisation":
            plt.scatter(x,y, color = "steelblue", linewidths = 0.1)
        else:
            plt.scatter(x,y, color = "sandybrown", linewidths = 0.1)
        plt.show()

    def outputDistribution(self,condition,distribution,label1): # bar chart
        xlabel = []
        for k in condition:
            xlabel.append(k)
        plt.title(label1)
        plt.xlabel("Output bins")
        plt.ylabel("Count")
        if label1 == "The distribution of the original outputs":
            rectangles = plt.bar(xlabel,distribution,color = 'steelblue')
        else:
            rectangles = plt.bar(xlabel,distribution,color = 'sandybrown')

        plt.xticks(rotation=45)
        plt.grid(color='grey', axis = 'y', linestyle=':', linewidth=1)
        self.autolabel(rectangles)
        plt.show()

    def compareDistribution(self,condition,output1,output2,label1,label2): # bar char
        xlabel = []
        for k in condition:
            xlabel.append(k)
        plt.title("Outputs distribution")
        plt.xlabel("Output")
        plt.ylabel("Count")
        plt.grid(color='grey', axis = 'y', linestyle=':', linewidth=1)
        bins = np.arange(len(output1))
        width = 0.4
        print(bins)
        plt.xticks(bins,xlabel,rotation=45)
        rectangles1 = plt.bar(bins-width/2,output1,width = width,color = 'steelblue', label = label1)
        rectangles2 = plt.bar(bins+width/2,output2,width = width,color = 'sandybrown', label = label2)
        plt.legend()
        self.autolabel(rectangles1)
        self.autolabel(rectangles2)
        plt.show()


    # plot true y value and predicted y value
    def y_pred_VS_y_true_2D(self,y_true,y_pred):
        plt.plot(y_true,y_true,color ="orange")
        plt.scatter(y_true,y_pred, marker="o")
        plt.title("Comparison between true y value and predicted y value ")
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.show()


    # plot true y value and predicted y value
    def y_pred_VS_y_true(self,y_true,y_pred):
        index =  np.arange(len(y_true))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(index, y_true)
        # Z = X**0 - Y**1
        Z = Y**1
        ax.plot_surface(X, Y ,Z, rstride=1, cstride=1, color='pink')
        ax.scatter(index,y_pred,y_true, marker="o",color='steelblue')
        ax.set_xlabel("Index")
        ax.set_ylabel("Predicted ouput")
        ax.set_zlabel("True ouput")
        plt.title("Comparison between true y value and predicted y value ")
        plt.show()

    def compareScore(self,average_score,scorelist,name):
        average_score_list = []
        lowest = min(scorelist)
        highest = max(scorelist)
        times = np.arange(1,len(scorelist)+1)
        for t in times:
            average_score_list.append(average_score)
        plt.plot(times,average_score_list,color='steelblue')
        plt.plot(times,scorelist,color='sandybrown')
        plt.title(name + " comparison between different models when using 10-fold validation")
        plt.xlabel("Times")
        plt.ylabel("Score")
        plt.legend(["average_score","score"])
        plt.grid(color='grey', linestyle=':', linewidth=1)
        plt.ylim(average_score/10,2*average_score-average_score/10)
        plt.xticks(times,('1','2','3','4','5','6','7','8','9','10'))
        plt.show()

    def compareScore2(self,average_score1,scorelist1,average_score2,scorelist2,name):
        average_score_list1 = []
        average_score_list2 = []
        average_score = (average_score1+average_score2)/2
        # lowest = min(min(scorelist1),min(scorelist2))
        # highest = max(max(scorelist1),max(scorelist2))
        times = np.arange(1,len(scorelist1)+1)
        for t in times:
            average_score_list1.append(average_score1)
        for t in times:
            average_score_list2.append(average_score2)
        plt.plot(times,average_score_list1,color='steelblue', label ='average '+name+' for validation dataset')
        plt.plot(times,scorelist1,color='sandybrown',label = name + ' for validation dataset')
        plt.plot(times,average_score_list2,color='mediumvioletred',label = 'average '+name + ' for test dataset')
        plt.plot(times,scorelist2,color='seagreen',label = name+' for test dataset')
        plt.title(name + " comparison between in validation dataset and test dataset")
        plt.xlabel("Times")
        plt.ylabel("Score")
        plt.grid(color='grey', linestyle=':', linewidth=1)
        plt.ylim(average_score/1.5,2*average_score-average_score/1.5)
        plt.xticks(times)
        plt.legend()
        plt.show()


    def featureScore(self,scorelist1,label1,scorelist2,label2,title):
        times = np.arange(1,len(scorelist1)+1)
        lowest = min(min(scorelist1),min(scorelist2))
        highest = max(max(scorelist1),max(scorelist2))
        plt.plot(times,scorelist1,color='steelblue',label=label1)
        plt.plot(times,scorelist2,color='sandybrown',label=label2)
        plt.title("Comparison of scores when the number of features changes" + "(" + title + ")")
        plt.xlabel("The number of features")
        plt.ylabel("Score")
        plt.grid(color='grey', linestyle=':', linewidth=1)
        plt.ylim(lowest-0.05,highest+0.05)
        plt.xticks(times)
        plt.legend()
        plt.show()

    def featureScore2(self,scorelist1,label1,scorelist2,label2,scorelist3,label3,scorelist4,label4,title):
        times = np.arange(1,len(scorelist1)+1)
        lowest = min(min(min(scorelist1),min(scorelist2)),min(min(scorelist3),min(scorelist4)))
        highest = max(max(max(scorelist1),max(scorelist2)),max(max(scorelist1),max(scorelist2)))
        plt.plot(times,scorelist1,color='steelblue',label=label1)
        plt.plot(times,scorelist2,color='sandybrown',label=label2)
        plt.plot(times,scorelist3,color='mediumvioletred',label=label3)
        plt.plot(times,scorelist4,color='seagreen',label=label4)
        plt.title("Comparison of scores when the number of features changes" + "(" + title + ")")
        plt.xlabel("The number of features")
        plt.ylabel("Score")
        plt.grid(color='grey', linestyle=':', linewidth=1)
        plt.ylim(lowest-0.05,highest+0.05)
        plt.xticks(times)
        plt.legend()
        plt.show()

    def autolabel(self,rectangles):
        (y_bottom, y_top) = plt.ylim()
        y_height = y_top - y_bottom
        for r in rectangles:
            height = r.get_height()
            p_height = (height / y_height)
            if p_height > 0.95:
                label_position = height + (y_height * 0.005)
            else:
                label_position = height + (y_height * 0.01)
            plt.text(r.get_x() + r.get_width()/2., label_position,'%d' % int(height), ha='center', va='bottom')

    def autotable(self,times,model,parameters,criteria):
        table = PrettyTable()
        table.add_column("Times",times)
        table.add_column("Model",model)
        for (k,v) in  parameters.items():
            table.add_column(k,v)
        for (k,v) in  criteria.items():
            table.add_column(k,v)
        print (table)
