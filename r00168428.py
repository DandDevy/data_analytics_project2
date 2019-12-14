#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
author Daniel
"""
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
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from io import StringIO
from sklearn import linear_model
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

print("Name: Daniel Ashcroft \nStudent ID: r00168428")
#Note: Not all the libararies above are necessarily needed for this project and not all 
#the libraries you need for this project are necessarily listed above.


""" Your Name and Your student ID: 
Daniel Ashcroft
r00168428
"""


# In[19]:



def task1():
    """
    The data needed to be encoded.
    I saved both datasets but I actually just reused the X as they are the same.
    The first model with y instead of loan was more accurate. They to use 'no' a lot more than needed.
    """
    print("--------task1--------")
    bank = pd.read_csv("bank.csv")
    task1_dataset1_csv_file_name = "task1_dataset1.csv"
    task1_dataset2_csv_file_name = "task1_dataset2.csv"
    
    
    # creates and saves the first data set

    dataset1 = bank[["age","job","poutcome","balance","default","y"]]  
    dataset1.to_csv(task1_dataset1_csv_file_name)
    print("dataset1 saved to file as", task1_dataset1_csv_file_name)

    # creates and saves the second data set
    
    dataset2 = bank[["age","job","poutcome","balance","default","loan"]]
    dataset2.to_csv(task1_dataset2_csv_file_name)
    print("dataset2 saved to file as", task1_dataset2_csv_file_name)

    
    

    #gets the data and shows it
    dataset1 = pd.read_csv(task1_dataset1_csv_file_name)
    dataset2 = pd.read_csv(task1_dataset2_csv_file_name)

    #which columns are categorical
    Xdataset_catg_cols = ["job","poutcome","default"]
    # all columns
    Xdataset_cols = ["age","job","poutcome","balance","default"]
    X = dataset1[Xdataset_cols]

    # this will encode the data from the columns with categorical data
    le = LabelEncoder()
    X[Xdataset_catg_cols] = X[Xdataset_catg_cols].apply(lambda col: le.fit_transform(col))

    # data set 1s y class atribute
    ydataset1 = dataset1['y']
    
    print("\n\nOur data set X value encoded:\n",X)
    print("\nOur y class atribute:\n",ydataset1)

    # using DecisionTreeClassifier for classification algorithm
    clf_y = DecisionTreeClassifier()
#     clf_y.fit(X.head(100),ydataset1.head(100))
    clf_y.fit(X,ydataset1)

    y_predictions = clf_y.predict(X.head(100))
    y_predictions_as_series = pd.Series(y_predictions)
    test_labels1 = ydataset1.head(100)

    res1 = accuracy_score(test_labels1, y_predictions_as_series)
    print("\n\naccuracy_score with first data set:\n",res1)
    #it only gives  NOs
    
    # data set 2s loan class attribute
    loan_dataset2 = dataset2['loan']

    clf_loan = DecisionTreeClassifier()
#     clf_loan.fit(X.head(100),loan_dataset2.head(100))
    clf_loan.fit(X,loan_dataset2)
    loan_predictions = clf_y.predict(X.head(100))
    loan_predictions_as_series = pd.Series(loan_predictions)

    test_labels2 = loan_dataset2.head(100)

    res1 = accuracy_score(test_labels2, loan_predictions_as_series)
    print("\n\naccuracy_score with second data set:\n",res1)
    
    
    print()
    print()
    print("Conclusion:")
    print("The data needed to be encoded")
    print("The first the data set with the y as there classifier has a slight higher accuracy")
    
    print()
    print("Possible error: We can see  for loan predictions especially that our models seem to use 'no' much more than 'yes' than needed")
    print("amount of value counts for y predictions:\n",y_predictions_as_series.value_counts())
    print("amount of value counts for y actual:\n",ydataset1.value_counts())
    print("amount of value counts for loan predictions:\n",loan_predictions_as_series.value_counts())
    print("amount of value counts for loan actual:\n",loan_dataset2.value_counts())
    

#0.8829709583950809 is less than 0.8398177434695097
# class attribute with y has a higher accuracy score
    
    

    print("--------end task1--------")






task1()


# In[5]:


def task2():
    '''
    used this post on medium: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
    I used the elbow method here to visualise which is the best
    The optimal number of clusters is 3
    '''
    print("--------task2--------")

    # for creating and saving the data set.

    bank = pd.read_csv("bank.csv")
    task2_dataset_csv_file_name = "task2_dataset.csv"
    task2_dataset_cols = ["age","marital"]
    dataset = bank[task2_dataset_cols]
    dataset.to_csv(task2_dataset_csv_file_name)
    print("dataset saved")

    # reads data, excluding the row number

    dataset = pd.read_csv(task2_dataset_csv_file_name)[task2_dataset_cols]
    print(dataset)

    # we are going to encode the marital categorical data 
    print(dataset.marital.unique())

    le = LabelEncoder()
    dataset.marital = le.fit_transform(dataset.marital)

    print(dataset.age.describe())
    Sum_of_squared_distances = []
    K = range(1,12)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(dataset)
        Sum_of_squared_distances.append(km.inertia_)


    plt.plot(K, Sum_of_squared_distances, 'x-')
    plt.xlabel('k the number of clusters')
    plt.ylabel('Sum of the squared distances')
    plt.title('Elbow Method For Optimal k. (Example found on medium reference and link in top and bottom comment of task)')
    plt.legend(["sum of sq dist for different k"])
    plt.show()

    print("We can see elbow at 3, the optimal number cluster")
    print("reference, used this post on medium: https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f")


    print("--------end task2--------")

task2()


# In[6]:


def task3():
    """
    Testing the accuracy with SVM for a range of bucket amounts
    """
    print("--------task3--------")

    # for creating and saving the data set.
    bank = pd.read_csv("bank.csv")
    task3_dataset_csv_file_name = "task3_dataset.csv"
    task3_dataset_cols = ["y","loan","bank_arg1"]
    dataset = bank[task3_dataset_cols]
    dataset.to_csv(task3_dataset_csv_file_name)
    print("dataset saved")

    # reads data, excluding the row number

    dataset = pd.read_csv(task3_dataset_csv_file_name)[task3_dataset_cols]
    dataset_catg_cols = ["y","loan"]

    # this will encode the data from the columns with categorical data
    le = LabelEncoder()
    dataset[dataset_catg_cols] = dataset[dataset_catg_cols].apply(lambda col: le.fit_transform(col))

    scores = []

    # testing with range of buckets
    range_number_of_buckets = range(2,15)
    for number_of_buckets in range_number_of_buckets:
        temp_dataset = dataset.copy()

        #bucketing
        temp_dataset["bank_arg1"] = pd.qcut(dataset["bank_arg1"],number_of_buckets,labels=False)
        # enc = KBinsDiscretizer(n_bins=10, encode='onehot')


        y = temp_dataset["bank_arg1"]
        X = temp_dataset[["y","loan"]]

        # X_binned = enc.fit_transform(X)

    
        clf = SVC()
        clf.set_params(kernel='linear')
        clf.fit(X.head(100),y.head(100))
        y_predictions = pd.Series(clf.predict(X.head(100)))
        score = accuracy_score(y.head(100), y_predictions)
        scores.append(score)
    #     print(scores)


    print(scores)
    plt.plot(range_number_of_buckets, scores, 'x-')
    plt.xlabel('number of buckets')
    plt.ylabel('accuracy score')
    plt.title('The accuracy score for different buckets')
    plt.legend(["accuracy score"])
    plt.show()

    print("The highest accuracy score is with 2 buckets")

    print("------end task3------")
task3()


# In[7]:


def task4():
    """
    It seems DecisionTreeClassifier has the worst accuracy and has a higher standard deviation
    SVM is very slow to fit
    KNeighborsClassifier, GaussianNB or RandomForestClassifier are good depending on what you want to do
    """
    print("--------task4--------")
    bank = pd.read_csv("bank.csv")
    task4_dataset_csv_file_name = "task4_dataset.csv"
    task4_dataset_cols = ["age","job","marital","education","loan","y"]
    dataset = bank[task4_dataset_cols]
    dataset.to_csv(task4_dataset_csv_file_name)
    print("dataset saved")
    # reads data, excluding the row number
    dataset = pd.read_csv(task4_dataset_csv_file_name)[task4_dataset_cols]

    # setting X
    X = dataset[["age","job","marital","education","loan"]]
    X_catg_cols = ["job","marital","education","loan"]

    # encoding X
    le = LabelEncoder()
    X[X_catg_cols] = X[X_catg_cols].apply(lambda col: le.fit_transform(col))
    print(dataset)
    print(X)
    amount = int(len(dataset.index)/25)
    X = X.head(amount)
    print(len(dataset.index))
    print(len(X.index))

    # setting y

    y = dataset['y']
    y = y.head(amount)
    
    # setting models
    models = []
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    models.append(('RF', RandomForestClassifier(max_depth=2, random_state=0)))
    # models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))

    print(models)
    
    #cross validation
    results = []
    names = []
    results_mean = []
    results_std = []
    for name, model in models:
        kfold = StratifiedKFold(n_splits=10, random_state=1)
        cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        results_mean.append(cv_results.mean())
        results_std.append(cv_results.std())
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

        
    
    plt.bar(names, results_mean)
    plt.xlabel('models')
    plt.ylabel('accuracy score')
    plt.title('The accuracy score for models')
    plt.show()

    plt.bar(names, results_std)
    plt.xlabel('models')
    plt.ylabel('standard deviation')
    plt.title('The standard deviation for models')
    plt.show()

    print("It seems DecisionTreeClassifier is the worst")
    print("SVM is very slow to fit")
    print("KNeighborsClassifier, GaussianNB or RandomForestClassifier are good depending on what you want to do")
    
    
    print("------end task4------")
    
task4()


# In[8]:


def task5():
    """
    Clusters and plots with kmeans with different k
    Altough the elbow method here recommends otherwise. Only with k is 5 that the last region gets properly considered as it's own cluster
    """
    print("--------task5--------")

    #creates data set

    bank = pd.read_csv("bank.csv")
    task5_dataset_csv_file_name = "task5_dataset.csv"
    task5_dataset_cols = ['bank_arg1',"bank_arg2"]
    dataset = bank[task5_dataset_cols]
    dataset.to_csv(task5_dataset_csv_file_name)
    print("dataset saved")

    dataset = pd.read_csv(task5_dataset_csv_file_name)[task5_dataset_cols]

    print(dataset)






    Sum_of_squared_distances = []

     # runs Kmeans for different number of clusters 
        
    K = range(1,6)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(dataset)

        cl_centers = km.cluster_centers_
        km_labels = km.labels_
        print("With k as ",k)
    #     for i, l in enumerate(km.labels_):

    #         plt.scatter(dataset["bank_arg1"].get(i), dataset["bank_arg2"].get(i), color=colours[l], marker=markers[l])
    
        title = 'scatter plot of bank_arg1 and bank_arg2 with ' + str(k) + " centroids"
        
        # plots by the kmeans labels for colours
        plt.scatter(dataset["bank_arg1"], dataset["bank_arg2"],c=km.labels_.astype(float))
        plt.scatter(cl_centers[:, 0], cl_centers[:, 1], c='red', marker='x')
        plt.xlabel('bank_arg1')
        plt.ylabel('bank_arg2')
        plt.title(title)
        plt.show()
        Sum_of_squared_distances.append(km.inertia_)

        
    plot_title = 'Elbow Method For Optimal k'
    plt.plot(K, Sum_of_squared_distances, 'x-')
    plt.xlabel('k the number of clusters')
    plt.ylabel('Sum of the squared distances')
    plt.title(plot_title)
    plt.legend()
    plt.show()


    print("Altough the elbow method recommends otherwise. Only with k is 5 that the last region gets properly considered as it's own cluster")
    print("------end task5------")
task5()


# In[9]:


def task6():
    """
    https://stackoverflow.com/questions/35097003/cross-validation-decision-trees-in-sklearn
    Here the machine learning algorthm overfits and produces a model that doesn't capture the tread of data. 
    I used cross validation to try to deal with overfitting

    """
    print("--------task6--------")

    # creates and saves the dataset
    bank = pd.read_csv("bank.csv")
    task6_dataset_csv_file_name = "task6_dataset.csv"
    task6_dataset_cols = ['housing',"balance",'y']
    task6_dataset_catg_cols = ['housing']
    dataset = bank[task6_dataset_cols]
    dataset.to_csv(task6_dataset_csv_file_name)
    print("dataset saved")

    dataset = pd.read_csv(task6_dataset_csv_file_name)[task6_dataset_cols]
    le = LabelEncoder()
    dataset[task6_dataset_catg_cols] = dataset[task6_dataset_catg_cols].apply(lambda col: le.fit_transform(col))

    print(dataset)
    X_train = dataset[['housing',"balance"]].head(10000)
    y_train = dataset['y'].head(10000)
    X_test = dataset[['housing',"balance"]].tail(1000)
    y_test = dataset['y'].tail(1000)
    clf = DecisionTreeClassifier()
    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred_train = clf.predict(X_train)
    training_score = accuracy_score(y_train, y_pred_train)
    print(training_score)

    y_pred_test = clf.predict(X_test)
    testing_score = accuracy_score(y_test, y_pred_test)
    print(testing_score)
    print("training_score and the testing_score are very different")



    # We can use cross validation for example

    parameters = {'max_depth':range(3,20)}
    clf_with_grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
    clf_with_grid_search.fit(X=X_train, y=y_train)
    tree_model = clf_with_grid_search.best_estimator_
    print (clf_with_grid_search.best_score_, clf_with_grid_search.best_params_) 

    y_pred_test = clf_with_grid_search.predict(X_test)
    testing_score = accuracy_score(y_test, y_pred_test)
    print(testing_score)


    depth = []
    for i in range(2,20):
        clf = DecisionTreeClassifier(max_depth=i)
        # Perform 7-fold cross validation 
        scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=7, n_jobs=4)
        depth.append((i,scores.mean()))
    print(depth)

    print("------end task6------")
task6()


# In[38]:


def task7():
    """
    RandomForestClassifier has a better score over DecisionTreeClassifier after using train_test_split
    """
    print("--------task7--------")
    # creates and saves the dataset
    bank = pd.read_csv("bank.csv")
    task7_dataset_csv_file_name = "task7_dataset.csv"
    task7_dataset_cols = ['loan',"balance",'y','bank_arg1']
    task7_dataset_catg_cols = ['loan','y']
    # task7_dataset_catg_cols = ['housing']
    dataset = bank[task7_dataset_cols]
    dataset.to_csv(task7_dataset_csv_file_name)
    print("dataset saved")

    dataset = pd.read_csv(task7_dataset_csv_file_name)[task7_dataset_cols]

    # encoding the categorical data
    le = LabelEncoder()
    dataset[task7_dataset_catg_cols] = dataset[task7_dataset_catg_cols].apply(lambda col: le.fit_transform(col))

    #cutting bank_arg1 into buckets
    dataset["bank_arg1"] = pd.qcut(dataset["bank_arg1"],2,labels=False)
    # enc = KBinsDiscretizer(n_bins=4, encode='onehot')
    # dataset = enc.fit_transform(dataset)

    X = dataset[['loan',"balance",'y']]
    Y = dataset['bank_arg1']

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
    print(X_train)

    # DecisionTreeClassifier
    clf_dtree = DecisionTreeClassifier()
    clf_dtree.fit(X_train,y_train)

    clf_dtree_pred = clf_dtree.predict(X_test)

    clf_dtree_score = accuracy_score(clf_dtree_pred, y_test)

    print("DecisionTreeClassifier score:")
    print(clf_dtree_score)

    #RandomForestClassifier

    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
    clf_rf.fit(X_train,y_train)

    clf_rf_pred = clf_rf.predict(X_test)

    clf_rf_score = accuracy_score(clf_rf_pred, y_test)

    print("RandomForestClassifier score")
    print(clf_rf_score)

    print("RandomForestClassifier has a better score")

    print("------end task7------")
task7()    

