# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:57:03 2019

@author: Arpita Chikkodi
"""

import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from tkinter import *
import time


window = Tk()
window.title("Student Performance Analysis")
window.geometry('700x400')

##Please change the path before execution
df = pd.read_excel('D:/GitHub/Student_Performance_Analysis/Data1ForAnalysis.xlsx')


#if even a single column of that row has nan value remove whole row(118-9=109)
df1 = df.dropna(how='any')
df1.isnull().sum()

res1 = StringVar()
res1.set('')
res2 = StringVar()
res2.set('')
res3 = StringVar()
res3.set('')
res4 = StringVar()
res4.set('')

out = StringVar()
out.set('')

out2 = StringVar()
out2.set('')
cie = 0

result = Label(window,text="Student SEE marks prediction").grid(row=0,column=1)
a = Label(window ,text='Quiz1').grid(row = 1,column = 0,padx=65)
b = Label(window ,text = "Quiz2").grid(row = 2,column = 0)
c = Label(window ,text = "Average Internal").grid(row = 3,column = 0)
d = Label(window ,text = "Assignment").grid(row = 4,column = 0)



quiz1 = Entry(window,textvariable=res1).grid(row = 1,column = 1)
quiz2 = Entry(window,textvariable=res2).grid(row = 2,column = 1)
finaltest = Entry(window,textvariable=res3).grid(row = 3,column = 1)
assignment = Entry(window,textvariable=res4).grid(row = 4,column = 1)

result = Label(window,textvariable=out).grid(row=6,column=1)
result = Label(window,textvariable=out2).grid(row=7,column=0)

def RegAlgo():
    df1 = df.dropna(how='any')
    X = df1.iloc[:118,5:9]
    Y = df1.iloc[:118,10]
    X_train,Y_train = X,Y
    X_train.size  
    Y_train.size
    # create linear regression model
    reg = linear_model.LinearRegression()
    print(reg)
    # train the model using the training sets 
    reg.fit(X_train, Y_train) 
    # regression coefficients 
    print('Coefficients: \n', reg.coef_) 
    predictors_test = []
    predictors_test.append(int(res1.get()))
    predictors_test.append(int(res2.get()))
    predictors_test.append(int(res3.get()))
    predictors_test.append(int(res4.get()))
    print(predictors_test)
    predictors_test1 = [np.array(predictors_test)]
    predicted = reg.predict(predictors_test1)
    result = Label(window,text="Predicted SEE marks =").grid(row=6,column=0)
    see = int(predicted)
    out.set(see)
    cie = sum(predictors_test)
    st2 = time.time()
    knnAlgo()
    et2 = time.time()
    print("Time taken by KNN classification algorithm is ",et2-st2)
    
def knnAlgo():
        data = df.dropna(how='any')
        #segregate the predictor variables
        predictors_train = data.iloc[:118,5:10]
        #segregate the target/ class variable
        target_train = data.iloc[:118,11]
        #instantiate the model with 1 neighbor
        nn = KNeighborsClassifier(n_neighbors = 1,metric='euclidean')
        #first train the model / classifier with the input data set
        model = nn.fit(predictors_train,target_train)
        print(model)
        predictors_test = []
        predictors_test.append(int(res1.get()))
        predictors_test.append(int(res2.get()))
        predictors_test.append(int(res3.get()))
        predictors_test.append(int(res4.get()))
        see = int(out.get())
        predictors_test.append(see)
        print(predictors_test)
        predictors_test1 = [np.array(predictors_test)]
        predicted = model.predict(predictors_test1)
        if(predicted == ['P']):
            out2.set("Result is Pass")
        else:
            out2.set("Result is Fail")


def CheckAlgo():
    if(res1.get() == '' or res2.get() == '' or res3.get() == '' or res4.get() == ''):
        out2.set("Enter the missing values!")
    elif(int(res1.get()) < 0 or int(res1.get()) > 10):
        out2.set("Error! Quiz can take values between 0 and 10 only")
    elif(int(res2.get()) < 0 or int(res2.get()) > 10):
        out2.set("Error! Quiz can take values between 0 and 10 only")
    elif(int(res3.get()) < 0 or int(res3.get()) > 50):
        out2.set("Error! Average internal marks should be between 0 and 50 only!")
    elif(int(res4.get()) < 0 or int(res4.get()) > 50):
        out2.set("Error! Assignment can take values between 0 and 30 only")
    else:
        #st1 = time.time()
        RegAlgo()
        #et1 = time.time()
        #print("Time taken by regression algorithm is ",et1-st1)
    
btn = Button(window ,text="Submit",command=CheckAlgo).grid(row=5,column=1)
window.mainloop()