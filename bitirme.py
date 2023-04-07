import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import datetime
import time

fonksiyonBasiSaat = 12
gun = fonksiyonBasiSaat * 60


def log(log):
    su_an = datetime.datetime.now()
    tarih_ve_saat = su_an.strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as dosya:
        dosya.write(tarih_ve_saat +"\t"+ log+ "\t\n")

ccard_data=pd.read_csv('creditcard.csv')

#separating the data for analysis
legit = ccard_data[ccard_data.Class==0]
fraud= ccard_data[ccard_data.Class==1]

#normal işlemlerin sayısını düşür. 
legit_sample= legit.sample(n=180000)  # random olarak alır.


#Concatenating two dataframes
new_dataset=pd.concat([legit_sample,fraud],axis=0) #axis 0 ile legit sampleın altına eklendi. axis 1 olsa yeni bir column olarak eklenecekti.


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaling = scaler.fit_transform(np.array(new_dataset['Amount']).reshape(-1, 1))
scaling2 = scaler.fit_transform(np.array(new_dataset['Time']).reshape(-1, 1))
new_dataset['Amount']=scaling
new_dataset['Time']=scaling2
X = new_dataset.drop('Class', axis=1)
y = new_dataset['Class']

from collections import Counter
from imblearn.combine import SMOTETomek

#Implementing the technique
smk = SMOTETomek(random_state=42)

# fit and apply the transform
X_smk, y_smk = smk.fit_resample(X, y)

#Make a train set dataframe for SMOTE
df_smote = X_smk
df_smote['Class']=y_smk

#Reset the index
df_smote['index']=[i for i in range(len(df_smote))]
df_smote = df_smote.set_index('index')


X_smk = X_smk.drop(['index','Class'],axis=1)

X=df_smote.drop(columns='Class', axis=1) #axis 1 çünkü column silcen
Y=df_smote['Class']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3, stratify=Y, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

def fonksiyon1():
    # Logistic Regression 

    model= LogisticRegression(max_iter=32000) # dataset boyutunu aşıyor. default iteration değeri 1000. 
    # training  the logistic regression model with traning data (X_train) 

    model.fit(X_train,Y_train) # X_train tüm featureları tutuyo. Y_train onlara karşılık gelen Label'ları tutuyor.
    #Accuracy Score
    #accuracy on training data
    X_train_prediction = model.predict(X_train)
    training_data_accuracy= accuracy_score(X_train_prediction, Y_train)

    log("Accuracy on Training data; "+ str(training_data_accuracy))

    #accuracy on test data
    X_test_prediction = model.predict(X_test)
    test_data_accuracy= accuracy_score(X_test_prediction, Y_test)

    log("Accuracy on Test data; "+str(test_data_accuracy))

def fonksiyon2():
    #Decision Tree Classification

    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, Y_train)
    y_pred=classifier.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    log(str(cm))
    accuracy_score(Y_test, y_pred)
    Accuracy_Decison = ((cm[0][0] + cm[1][1]) / cm.sum()) *100
    log("dec tree Accuracy_Decison    : "+ str(Accuracy_Decison))
    Error_rate = ((cm[0][1] + cm[1][0]) / cm.sum()) *100
    log("Error_rate  : "+str(Error_rate))


def fonksiyon3():
    #Random Tree Classification

    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, Y_train)
    y_pred=classifier.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    log(str(cm))
    accuracy_score(Y_test, y_pred)
    Accuracy_Decison = ((cm[0][0] + cm[1][1]) / cm.sum()) *100
    log("ranDOM foresT Accuracy_Decison    : "+ str(Accuracy_Decison))
    Error_rate = ((cm[0][1] + cm[1][0]) / cm.sum()) *100
    log("Error_rate  : "+str(Error_rate))

def fonksiyon4():
    # KNN Classifier

    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(Y_test, y_pred)
    log(str(cm))
    accuracy_score(Y_test, y_pred)
    Accuracy_Decison = ((cm[0][0] + cm[1][1]) / cm.sum()) *100
    log(" knn Accuracy_Decison    : "+str( Accuracy_Decison))
    Error_rate = ((cm[0][1] + cm[1][0]) / cm.sum()) *100
    log("Error_rate  : "+str( Error_rate))

def fonksiyon5():
    #KMEANS
    kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
    kmeans.fit(X_train)

    predictions = kmeans.predict(X_test)

    pred_fraud = np.where(predictions == 1)[0]
    real_fraud = np.where(Y_test == 1)[0]
    false_pos = len(np.setdiff1d(pred_fraud, real_fraud))

    pred_good = np.where(predictions == 0)[0]
    real_good = np.where(Y_test == 0)[0]
    false_neg = len(np.setdiff1d(pred_good, real_good))

    false_neg_rate = false_neg/(false_pos+false_neg)

    accuracy = (len(X_test) - (false_neg + false_pos)) / len(X_test)

    log("Accuracy:" + str(accuracy))
    log("False negative rate (with respect to misclassifications): "+ str(false_neg_rate))
    log("False negative rate (with respect to all the data): "+ str((false_neg / len(predictions))))
    log("False negatives, false positives, mispredictions:"+str (false_neg)+" "+str(false_pos)+" " +str(false_neg + false_pos))
    log("Total test data points:"+ str(len(X_test)))




	
baslangic_zamani = datetime.datetime.now()



flag1=True
flag2=True
flag3=True
flag4=True
flag5=True

while True:
    gecen_zaman = datetime.datetime.now() - baslangic_zamani
    dakika = gecen_zaman.total_seconds() / 60
	
	
    if dakika<=gun:
        if flag1==True:
            log("fonk1 basladi")
            flag1=False
        fonksiyon1()
		
    elif dakika>gun and dakika<=(2*gun):
        if flag2==True:
            log("fonk2 basladi")
            flag2=False
        fonksiyon2()
		
    elif dakika>(2*gun) and dakika<=(3*gun):
        if flag3==True:
            log("fonk3 basladi")
            flag3=False
        fonksiyon3()
	    
		
    elif dakika>(3*gun) and dakika<=(4*gun):
        if flag4==True:
            log("fonk4 basladi")
            flag4=False
        fonksiyon4()
		
    elif dakika>(4*gun) and dakika<=(5*gun):
        if flag5==True:
            log("fonk5 basladi")
            flag5=False
        fonksiyon5()
		

    elif dakika>(5*gun):
        log("Hepsi Bitti")
        exit()