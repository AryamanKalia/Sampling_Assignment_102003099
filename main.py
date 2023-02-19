from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import math
data = pd.read_csv('Creditcard_data.csv')
x=data.iloc[:,:-1]
y=data.iloc[:,-1]
print('previous shape',x.shape,y.shape)
sm = SMOTE(random_state=42)
x_res, y_res = sm.fit_resample(x, y)
print(x_res.shape,y_res.shape)
print('counter of 0',y_res.value_counts()[0])
print('counter of 1',y_res.value_counts()[1])
from sklearn.model_selection import train_test_split
oversampled_data=pd.DataFrame(x_res)
oversampled_data['Class']=y_res

print(oversampled_data.shape)

train_set,test_set=train_test_split(oversampled_data,test_size=0.2,random_state=42)#now use train_set to train the model and dont use test_set

#implementing random sampling here
n=round(train_set.shape[0]*0.9)
random_sample_train=train_set.sample(n=n,random_state=42)


original_y=test_set.iloc[:,-1]#y labels of the test set
test_set=test_set.iloc[:,:30]#x labels of the test set
print('some stuff',random_sample_train.shape,test_set.shape,train_set.shape)

#training logistic regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#training random forest model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#training naive bayes model
from sklearn.naive_bayes import GaussianNB

#training decision tree model
from sklearn.tree import DecisionTreeClassifier
print('random sampling:')
clf=LogisticRegression(max_iter=10000)
clf.fit(random_sample_train.iloc[:,:30],random_sample_train.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for logistic  ', accuracy_score(original_y,y_pred))
# random_forest(random_sample_train)
clf=RandomForestClassifier()
clf.fit(random_sample_train.iloc[:,:30],random_sample_train.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for random forest' ,accuracy_score(original_y,y_pred))
# svm(random_sample_train)
clf=SVC()
clf.fit(random_sample_train.iloc[:,:30],random_sample_train.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for svm  ', accuracy_score(original_y,y_pred))
#naive_bayes(random_sample_train)
clf=GaussianNB()
clf.fit(random_sample_train.iloc[:,:30],random_sample_train.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for naive bayes ', accuracy_score(original_y,y_pred))
#decision_tree(random_sample_train)
clf=DecisionTreeClassifier()
clf.fit(random_sample_train.iloc[:,:30],random_sample_train.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for decision tree ', accuracy_score(original_y,y_pred))
#using structured sampling

# Calculate the number of rows in the dataset
n = len(train_set)

# Set the sampling interval "k" as the square root of the number of rows in the dataset
k = 3

# Select every "k" row starting from a random index in the dataset
structured_sample = train_set.iloc[::k]
print('structured sampling:',structured_sample.shape)

# logistic_regression(structured_sample)
clf=LogisticRegression(max_iter=10000)
clf.fit(structured_sample.iloc[:,:30],structured_sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for logistic  ', accuracy_score(original_y,y_pred))
# random_forest(structured_sample)
clf=RandomForestClassifier()
clf.fit(structured_sample.iloc[:,:30],structured_sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for random forest' ,accuracy_score(original_y,y_pred))
# svm(structured_sample)
clf=SVC()
clf.fit(structured_sample.iloc[:,:30],structured_sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for svm  ', accuracy_score(original_y,y_pred))
# naive_bayes(structured_sample)
clf=GaussianNB()
clf.fit(structured_sample.iloc[:,:30],structured_sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for naive bayes ', accuracy_score(original_y,y_pred))
# decision_tree(structured_sample)
clf=DecisionTreeClassifier()
clf.fit(structured_sample.iloc[:,:30],structured_sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for decision tree ', accuracy_score(original_y,y_pred))
#using stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
X=train_set.iloc[:,:30]
Y=train_set.iloc[:,-1]

print(train_set.shape)

for train_index, test_index in split.split(X,Y):

 strat_train_set = train_set.iloc[train_index,:]
 strat_test_set = train_set.iloc[test_index,:]

print('data for stratified sampling:')
# logistic_regression(strat_train_set)
# random_forest(strat_train_set)
# svm(strat_train_set)
# naive_bayes(strat_train_set)
# decision_tree(strat_train_set)
# logistic_regression(structured_sample)
clf=LogisticRegression(max_iter=10000)
clf.fit(strat_train_set.iloc[:,:30],strat_train_set.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for logistic  ', accuracy_score(original_y,y_pred))
# random_forest(strat_train_set)
clf=RandomForestClassifier()
clf.fit(strat_train_set.iloc[:,:30],strat_train_set.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for random forest' ,accuracy_score(original_y,y_pred))
# svm(strat_train_set)
clf=SVC()
clf.fit(strat_train_set.iloc[:,:30],strat_train_set.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for svm  ', accuracy_score(original_y,y_pred))
# naive_bayes(strat_train_set)
clf=GaussianNB()
clf.fit(strat_train_set.iloc[:,:30],strat_train_set.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for naive bayes ', accuracy_score(original_y,y_pred))
# decision_tree(strat_train_set)
clf=DecisionTreeClassifier()
clf.fit(strat_train_set.iloc[:,:30],strat_train_set.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for decision tree ', accuracy_score(original_y,y_pred))
#using cluster sampling
def get_clustered_Sample(df, n_per_cluster, num_select_clusters):
    N = len(df)
    K = int(N/n_per_cluster)
    data = None
    for k in range(K):
        sample_k = df.sample(n_per_cluster)
        sample_k["cluster"] = np.repeat(k,len(sample_k))
        df = df.drop(index = sample_k.index)
        data = pd.concat([data,sample_k],axis = 0)

    random_chosen_clusters = np.random.randint(0,K,size = num_select_clusters)
    samples = data[data.cluster.isin(random_chosen_clusters)]
    return(samples)

sample = get_clustered_Sample(df = train_set, n_per_cluster = 100, num_select_clusters = 20)
sample=sample.iloc[:,:31]
print('cluster sampling:')
# logistic_regression(sample)
# random_forest(sample)
# svm(sample)
# naive_bayes(sample)
# decision_tree(sample)
clf=LogisticRegression(max_iter=10000)
clf.fit(sample.iloc[:,:30],sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for logistic  ', accuracy_score(original_y,y_pred))
# random_forest(sample)
clf=RandomForestClassifier()
clf.fit(sample.iloc[:,:30],sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for random forest' ,accuracy_score(original_y,y_pred))
# svm(sample)
clf=SVC()
clf.fit(sample.iloc[:,:30],sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for svm  ', accuracy_score(original_y,y_pred))
# naive_bayes(sample)
clf=GaussianNB()
clf.fit(sample.iloc[:,:30],sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for naive bayes ', accuracy_score(original_y,y_pred))
# decision_tree(sample)
clf=DecisionTreeClassifier()
clf.fit(sample.iloc[:,:30],sample.iloc[:,-1])
y_pred=clf.predict(test_set)
print('accuracy score for decision tree ', accuracy_score(original_y,y_pred))