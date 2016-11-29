import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_Train=pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Loan Prediction/X_Train.csv')
Y_Train=pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Loan Prediction/Y_Train.csv')
X_Test=pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Loan Prediction/X_Test.csv')
Y_Test=pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Loan Prediction/Y_Test.csv')

#print (Y_Train.head())


X_Train[X_Train.dtypes[(X_Train.dtypes=="float64")|(X_Train.dtypes=="int64")].index.values].hist(figsize=[15,11],bins=50)
#plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_Train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']],Y_Train.values.ravel())

from sklearn.metrics import accuracy_score
print (accuracy_score(Y_Test,knn.predict(X_Test[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']])))

print (Y_Train.Target.value_counts()/Y_Train.Target.count())
print (Y_Test.Target.value_counts()/Y_Test.Target.count())

# Here, we're getting better accuracy in our test model than our prediction model?
    # this could be happening b/c of some insignificant variable with a larger range that will be dominating the function
    # we can remove this problem by scaling down all the features to a same range (using MinMaxScalar in sklearn)

from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()

X_Train_minmax = min_max.fit_transform(X_Train[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']])
X_Test_minmax = min_max.fit_transform(X_Test[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']])
    # now we're done scaling
knn.fit(X_Train_minmax,Y_Train.values.ravel())
print (accuracy_score(Y_Test,knn.predict(X_Test_minmax)))

