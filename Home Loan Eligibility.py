import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/nickwalker/Desktop/Data Sets/Home Loan Eligibility Train.csv')
#print (df.head(10))
#print (df.describe())
#print (df['Property_Area'].value_counts())
#df['ApplicantIncome'].hist(bins=50)
#df.boxplot(column='LoanAmount', by = ['Education','Self_Employed'])
#plt.show()


"""temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda
                       x: x.map({'Y':1, 'N':0}).mean())


print ("Frequency Table for Credit History:")
print (temp1)
print ("Probability of getting loan for each Credit History Class:")
print (temp2)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(221)
    # subplot(121) represents where the graphs lie (so its a 1x2 grid, 1st subplot) 
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(222)
ax2.set_xlabel('Credit History')
ax2.set_ylabel('Probability of Getting Loan')
ax2.set_title("Probability of Getting Loan by Credit History")
temp2.plot(kind='bar')
    # this shows the chances of getting a loan are eight-fold if the applicant has a valid credit history

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'], df['Gender'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

plt.show()"""


print(df.apply(lambda x: sum(x.isnull()),axis=0))

df['LoanAmount'].fillna(df['LoanAmount'].mean(),inplace=True)


#print (df['Self_Employed'].value_counts())
    #shows us that 86% of values for this are "No"
df['Self_Employed'].fillna('No',inplace=True)
    #so fill up the empty ones with "No" since you'll have an ~86% chance of being right


table = df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.sum)
#define function to return value of this pivot table
def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]


# Treating Extreme Values in the Distribution of LoanAmount and ApplicantIncome

#Starting with LoanAmount
    #thinking about it intuitively, some people might apply for high value loans due to specific needs
        # So instead of treating these values as outliers, lets try a log transformation to nullify their effect
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
        # now the distribution looks much closer to a normal dist. 


# Next is ApplicantIncome
    # intuitively, some applicants with low income can have coapplicants with hight income
        # so we create a total income column by combining these two values and taking a log transformation fo that
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
#plt.show()


# scikit learn is the most commonly used library in Python for predictive modeling 
    # but it requires all inputs to be numeric, so we have to convert all our categoric variables into numeric:

"""from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()

for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes

This doesn't work but was used in the example code


#import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold              #for K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics"""

#initializing and fitting a kNN model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome','LoanAmount', 'Loan_Amount_Term','Credit_History']],Y_train)


#checking the performance of our model on the testing data set
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]))


