# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 16:57:03 2019

@author: Arpita Chikkodi
"""
# Python code to illustrate  
# regression using data set 
#import matplotlib 

#There are two types of backends: user interface backends (for use in pygtk, wxpython, tkinter, qt4, or macosx; also referred to as “interactive backends”) and hardcopy backends to make image files (PNG, SVG, PDF, PS; also referred to as “non-interactive backends”).
#GTKAgg	Agg rendering to a GTK 2.x canvas (requires PyGTK and pycairo or cairocffi; Python2 only)
#AGG	png	raster graphics – high quality images using the Anti-Grain Geometry engine

#matplotlib.use('GTKAgg')

import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##Please change the path before execution
df = pd.read_excel('D:/MLProject/Data1ForAnalysis.xlsx')

df.isnull().sum()
df.notnull().tail()

#display nan values from the data
df[df['TEST1'].isnull()]

df[df['QUIZ1'].isnull()]

df[df['TEST2'].isnull()]

df[df['QUIZ2'].isnull()]

df[df['Assignment'].isnull()]

df[df['Result'].isnull()]

#if even a single column of that row has nan value remove whole row(118-9=109)
df1 = df.dropna(how='any')
df1.isnull().sum()


#CIE
X = df.iloc[:118,5:9]
X.shape
#SEE
Y = df.iloc[:118,10]
Y.shape


'''
plt.hist(Y,color='black') 
plt.title('SEE MARKS') 
plt.xlabel('SEE')'''

df1 = df.dropna(how='any')

X = df1.iloc[:118,5:9]
Y = df1.iloc[:118,10]

#To remove outliers for SEE the below commented code is used since the outliers are 
#not affecting the result or deviating, no need to remove them but we can analyse 
#how outliers can be removed from below commented code
##Y
'''iq=iqr(Y)
Q1 = np.quantile(Y,.25)
Q2 = np.quantile(Y,.50)
Q3 = np.quantile(Y,.75)
up = Q3+(1.5*iq)

lw = Q1 - (1.5*iq)

print("Upper whisker ",up)
print("Lower whisker ",lw)

#print(Q1)
#print(Q3)
#print(1.5*iq)

#Median
print("Median ",Q2)

sd = np.std(Y, axis=0)
print("Standard deviation ",sd)

#consider values between 32.5 and 92.5
Y = [i for i in Y if (i < up)]
Y = [i for i in Y if (i > lw)]
#print(Y)
#X.drop(100)


plt.boxplot(Y) 
plt.title('SEE MARKS DISTRIBUTION SHOWING OUTLIERS') 
plt.xlabel('SEE')
'''


X = df1.iloc[:118,5:9]
Y = df1.iloc[:118,10]


##test_size = 0.3 means 30% of original data to be considered as test data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=111)

print(X_train)
X_train.size  #70% of 109 = 76
len(X_train)


print(X_test)
X_test.size  #30% of 109 = 33
len(X_test)

# create linear regression object 
reg = linear_model.LinearRegression() 
print(reg)
# train the model using the training sets 
reg.fit(X_train, Y_train) 

# regression coefficients 
print('Coefficients: \n', reg.coef_) 
  
#variance score: 1 means perfect prediction 
print("Accuracy is ",reg.score(X_test,Y_test))