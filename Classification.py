# -*- coding: utf-8 -*-
"""
Created on Fri OCT  8 20:49:12 2019

@author: Arpita
"""



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pylab as pl
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import iqr

##Please change the path before execution
url  = "D:/MLProject/Data1ForAnalysis.xlsx"

class Classification(object):
               
    def preprocessing(self):
        #print(odata)
        print("Original data : ",odata.shape)
        odata[odata['Result'].isnull()]
        #if even a single column of that row has nan value remove whole row(118-9=109)
        self.data = odata.dropna(how='any')
        print("Features considered after preprocessing :",self.data.shape)
        print(self.data)
        #segregate the predictor variables
        self.predictors = self.data.iloc[:118,5:11]
        print("Predictor variables : ",self.predictors.shape)
        print(self.predictors)
        #segregate the target/ class variable
        self.target = self.data.iloc[:118,11]
        print("Target variable : ",self.target.shape)
        print(self.target)
        ##test_size = 0.3 means 30% of original data to be considered as test data
        self.predictors_train,self.predictors_test,self.target_train,self.target_test = train_test_split(self.predictors,self.target,
                                                                             test_size=0.3,random_state=10)
        #70% is training data i.e 70% of 109 is 76
        self.predictors_train.size  #76*6attributes
        print("Training data size : ",len(self.predictors_train)) 
        print(self.predictors_train)
        #30% is testing data i.e 30% of 109 is 36
        print(self.predictors_test)
        print("Test data size :",self.predictors_test.size)   #36*2
        print("Test data size : ",len(self.predictors_test))
        print(self.target_train)
        self.target_train.size  #32.6*1attribute
        print(len(self.target_train))
        print(self.target_test)
        self.target_test.size  #32.6*1attributes
        print(len(self.target_test))
        
        
    def preprocessing1(self):
        odata[odata['Result'].isnull()]
        self.data = odata.dropna(how='any')
        #segregate the predictor variables
        self.predictors = self.data.iloc[:118,5:11]
        #segregate the target/ class variable
        self.target = self.data.iloc[:118,11]
        #self.target.shape
        ##test_size = 0.3 means 30% of original data to be considered as test data
        self.predictors_train,self.predictors_test,self.target_train,self.target_test = train_test_split(self.predictors,self.target,
                                                                             test_size=0.3,random_state=111)

 
    def plotquiz(self):
        q1 = odata['QUIZ1']
        q2 = odata['QUIZ2']
        #q3 = odata['FinalTest']
        #qrange = [*range(1,119,1)]
        #print(qrange)
        print(q1.size)
        print(q2.size)
        #plt.scatter( qrange,q1, color='r')
        #plt.scatter(qrange, q2, color='g')
        plt.scatter(q1,q2,color='b')
        plt.xlabel('Quiz 1')
        plt.ylabel('Quiz 2')
        plt.show()
        
    def plotassign(self):
        #plt.boxplot(odata['FinalTest'])
        plt.hist(odata['Assignment'],orientation='vertical',color='blue')
        #plt.hist(odata['Assignment'],orientation='vertical',color='blue')
        plt.title("Average Internal Marks")
        plt.xlabel('Internal marks')
        
        
    def knnAlgo(self):
        #instantiate the model with 1 neighbor
        self.nn = KNeighborsClassifier(n_neighbors = 1,metric='euclidean')
        #first train the model / classifier with the input data set, training part of it
        self.model = self.nn.fit(self.predictors_train,self.target_train)
        print(self.model)
        #to check the prediction accuracy
        #print(self.nn.score(self.predictors_test,self.target_test))
        #print(self.model.predict([3,7,10,12,32,0]))
        predicted = self.model.predict(self.predictors_test)
        expected = self.target_test
        matches = (predicted == expected)
        #print(matches.sum())
        #print(len(matches))
        print("Accuracy is ",matches.sum() / float(len(matches)))
        print(metrics.classification_report(expected, predicted))

    def plotSEECIE(self):
        plt.scatter(self.data['SEE'],self.data['Assignment'],color='blue')
        
    def outlierdetection(self):
        # Plot CIE AND SEE to identify outliers
        #X = odata['CIE']
        Y= odata['SEE']
        #plt.boxplot(X)
        plt.boxplot(Y)
        #plt.hist(X,color='black') 
        #plt.title('CIE MARKS')
        #plt.xlabel('CIE')
        #plt.hist(Y,color='black')
        plt.title('SEE MARKS')
        plt.xlabel('SEE')
        ##X
        '''print("For CIE")
        iq=iqr(X)
        Q1 = np.quantile(X,.25)
        Q2 = np.quantile(X,.50)
        Q3 = np.quantile(X,.75)
        up = Q3+(1.5*iq)
        lw = Q1 - (1.5*iq)
        print("Upper whisker is ",up)
        print("Lower whisker is ",lw)
        #Median
        print("Q3 is ",Q3)
        print("Q2/Median is ",Q2)
        print("Q1 is ",Q1)
        print("IQR is ",1.5*iq)
        sd = np.std(X, axis=0)
        X = [i for i in X if (i < up)]
        X = [i for i in X if (i > lw)]
        print("CIE marks after removing outliers")
        print(X)'''
        ##Y
        print("For SEE")
        iq=iqr(Y)
        Q1 = np.quantile(Y,.25)
        Q2 = np.quantile(Y,.50)
        Q3 = np.quantile(Y,.75)
        up = Q3+(1.5*iq)
        lw = Q1 - (1.5*iq)
        print("Upper whisker is ",up)
        print("Lower whisker is ",lw)
        #Median
        print("Q3 is ",Q3)
        print("Q2/Median is ",Q2)
        print("Q1 is ",Q1)
        print("IQR is ",1.5*iq)
        sd = np.std(Y, axis=0)
        print("Standard deviation ",sd)
        #consider values between 32.5 and 92.5
        Y = [i for i in Y if (i < up)]
        Y = [i for i in Y if (i > lw)]
        #5Y.drop[100]
        print(Y)
        
        
    def NaiveBayesAlgo(self):
        #self.predictors_train,self.predictors_test,self.target_train,self.target_test=train_test_split(self.predictors,self.target,test_size=0.3,random_state=123)
        self.gnb=GaussianNB()
        self.model=self.gnb.fit(self.predictors_train,self.target_train)
        #print(self.model)
        #first train the model/classifier with input dataset training set of it
        #print(self.gnb.score(self.predictors_test,self.target_test))
        
        predicted = self.model.predict(self.predictors_test)
        expected = self.target_test
        matches = (predicted == expected)
        #print(matches.sum())
        #print(len(matches))
        print("Accuracy is ",matches.sum() / float(len(matches)))
        print(metrics.classification_report(expected, predicted))

odata = pd.read_excel(url)

c1 = Classification()

c1.preprocessing()
#c1.preprocessing1()

c1.knnAlgo()

#c1.NaiveBayesAlgo()

#c1.plotquiz()
#c1.plotassign()
#c1.plotSEECIE()
#c1.outlierdetection()