from pandas import read_csv
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from sklearn import linear_model
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#Note: Not all the libararies above are necessarily needed for this project and not all 
#the libraries you need for this project are necessarily listed above.


""" Your Name and Your student ID: 
Daniel Ashcroft
r00168428
"""

def task1():
    print("--------task1--------")
    bank = pd.read_csv("bank.csv")
    task1_dataset1_csv_file_name = "task1_dataset1.csv"
    task1_dataset2_csv_file_name = "task1_dataset2.csv"
    # creates and saves the first data set
    #     dataset1 = bank[["age","job","poutcome","balance","default","y"]]
    #     dataset1.to_csv(task1_dataset1_csv_file_name)
    #     print("dataset1 saved to file as", task1_dataset1_csv_file_name)

    # creates and saves the second data set
    #     dataset2 = bank[["age","job","poutcome","balance","default","loan"]]
    #     dataset2.to_csv(task1_dataset2_csv_file_name)
    #     print("dataset2 saved to file as", task1_dataset2_csv_file_name)



    dataset1 = pd.read_csv(task1_dataset1_csv_file_name)
    print(dataset1)
    dataset2 = pd.read_csv(task1_dataset2_csv_file_name)
    # print(dataset2)
    # print(dataset2["loan"])

    Xdateset1 = dataset1.loc[:, dataset1.columns != 'y']
    ydataset1 = dataset1['y']
    print(Xdateset1)
    print(ydataset1)
    clf = SVC()
    clf.set_params(kernel='linear')
    clf.fit(Xdateset1,ydataset1)




task1()
    
    
def task2():
    print("task2")

    
    
    
    
def task3():
    print("task3")

    
    
    
    
def task4():
    print("task4")

    
    
    
    
def task5():
    print("task5")

    
    
    
def task6():
    print("task6")
    
    
    
    
def task7():
    print("task7")

    
    
    
    
    
    