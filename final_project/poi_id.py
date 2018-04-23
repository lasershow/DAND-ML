#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Machine Learning
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import svm, datasets, cross_validation
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from numpy import mean

import pandas as pd
import numpy as np
import matplotlib.pyplot

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi', 'bonus', 'exercised_stock_options', 'from_messages', 'other', 'salary', 'total_payments', 'fraction_from_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Allocation across classes (POI/non-POI)
poi = 0
for person in data_dict:
    if data_dict[person]['poi'] == True:
       poi += 1
print("Total number of poi: %i" % poi)
print("Total number of non-poi: %i" % (len(data_dict) - poi))

# Transpose
df = pd.DataFrame(data_dict).T

# total number of data points
print("total number of data points: %i" % df.shape[0])

# for missing value
df = df.where(df != "NaN", -1)

df["bonus"] = df["bonus"].astype('int64')
df["deferral_payments"] = df["deferral_payments"].astype('int64')
df["deferred_income"] = df["deferred_income"].astype('int64')
df["director_fees"] = df["director_fees"].astype('int64')
# df["email_address"] = df["email_address"].astype('')
df["exercised_stock_options"] = df["exercised_stock_options"].astype('int64')
df["expenses"] = df["expenses"].astype('int64')
df["from_messages"] = df["from_messages"].astype('int64')
df["from_poi_to_this_person"] = df["from_poi_to_this_person"].astype('int64')
df["from_this_person_to_poi"] = df["from_this_person_to_poi"].astype('int64')
df["loan_advances"] = df["loan_advances"].astype('int64')
df["long_term_incentive"] = df["long_term_incentive"].astype('int64')
df["other"] = df["other"].astype('int64')
df["poi"] = df["poi"].astype('bool')
df["restricted_stock"] = df["restricted_stock"].astype('int64')
df["restricted_stock_deferred"] = df["restricted_stock_deferred"].astype('int64')
df["salary"] = df["salary"].astype('int64')
df["shared_receipt_with_poi"] = df["shared_receipt_with_poi"].astype('int64')
df["to_messages"] = df["to_messages"].astype('int64')
df["total_payments"] = df["total_payments"].astype('int64')
df["total_stock_value"] = df["total_stock_value"].astype('int64')

# for missing values
df = df.where(df != -1, pd.np.nan)

columns = ["bonus","deferral_payments","deferred_income","director_fees","email_address","exercised_stock_options","expenses","from_messages","from_poi_to_this_person","from_this_person_to_poi","loan_advances","long_term_incentive","other","poi","restricted_stock","restricted_stock_deferred","salary","shared_receipt_with_poi","to_messages","total_payments","total_stock_value"]

for column in df:
    if column not in ("poi", "email_address"):
        df[column] = df[column].fillna(df[column].mean())

### Task 2: Remove outliers

# plotOutliers
def plotOutliers(data_set, feature_x, feature_y):
    """
    This function takes a dict, 2 strings, and shows a 2d plot of 2 features
    """
    matplotlib.pyplot.scatter(data_set[feature_x], data_set[feature_y])
    matplotlib.pyplot.xlabel(feature_x)
    matplotlib.pyplot.ylabel(feature_y)
    matplotlib.pyplot.show()

#Visualize data to identify outliers
# print(plotOutliers(df, 'total_payments', 'total_stock_value'))
# print(plotOutliers(df, 'from_poi_to_this_person', 'from_this_person_to_poi'))
# print(plotOutliers(df, 'salary', 'bonus'))
# print(plotOutliers(df, 'total_payments', 'other'))


# ------
# remove outliers by IsolationForest methods
X = df.drop(['email_address','poi'], axis=1)
y = df['poi']

from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, max_samples=100)
clf.fit(X)
y_pred = clf.predict(X)
# non_outlier 1, outlier -1
non_outlier = 1
predicted_index = np.where(y_pred == non_outlier)

df = df.iloc[predicted_index]
# end: remove outliers by IsolationForest methods
# ------

# remove outlier from visualize
df = df[df.from_messages < 4000 ]
df = df[df.exercised_stock_options < 1e7 ]
df = df[df.deferred_income > -2000000 ]
df = df[df.from_poi_to_this_person < 300 ]
df = df[df.loan_advances > 3e7 ]
df = df[df.restricted_stock_deferred > -100000 ]
df = df[df.to_messages < 10000 ]
df = df[df.restricted_stock < 0.6e7 ]

# manual detect outlier
df = df.drop("THE TRAVEL AGENCY IN THE PARK")

# Visualize data after remove outliers
# print(plotOutliers(df, 'total_payments', 'total_stock_value'))
# print(plotOutliers(df, 'from_poi_to_this_person', 'from_this_person_to_poi'))
# print(plotOutliers(df, 'salary', 'bonus'))
# print(plotOutliers(df, 'total_payments', 'other'))

### Task 3: Create new feature(s)
fraction_from_poi =[]
fraction_to_poi =[]
for key, row in df.iterrows():
    fraction_from_poi.append(row["from_poi_to_this_person"] / row["to_messages"])
    fraction_to_poi.append(row["from_this_person_to_poi"] / row["from_messages"])

df['fraction_from_poi'] = fraction_from_poi
df['fraction_to_poi'] = fraction_to_poi

"""
I do feature selection, using domain knowledge.For example, email does not affect fraud.
Furthermore, I selected features using RF importance.
Results of RF importance My choice, fraction_from_poi, was chosen as important.

`ex`

y = df["poi"]
X = df.drop("poi",axis=1)
selector = RFE(RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=7)
selector.fit(X, y)
print selector.support_
print selector.ranking_

X_selected = selector.transform(X)
print "X.shape={}, X_selected.shape={}".format(X.shape, X_selected.shape)
"""


### Store to my_dataset for easy export below.
# data_dict = df.to_dict
my_dataset = df.T.to_dict("dict")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
features_list_X = ['bonus','deferred_income','exercised_stock_options','expenses',
                   'salary','total_payments','fraction_from_poi']

y = df['poi']
X = df[features_list_X]

"""
In machine learning, model validation is referred to as the process where a trained model is evaluated with a testing data set.
The testing data set is a separate portion of the same data set from which the training set is derived.
The main purpose of using the testing data set is to test the generalization ability of a trained model (Alpaydin 2010).

Validating data is extremely important.
The reason being is that at times data is not always perfect there can be imbalance within it which leads to skewed or biased results.
"""

### http://scikit-learn.org/stable/modules/pipeline.html
def evaluate_clf(grid_search, features, labels, params, iters=100):
    acc = []
    pre = []
    recall = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        acc = acc + [accuracy_score(labels_test, predictions)]
        pre = pre + [precision_score(labels_test, predictions)]
        recall = recall + [recall_score(labels_test, predictions)]
    print "accuracy: {}".format(mean(acc))
    print "precision: {}".format(mean(pre))
    print "recall:    {}".format(mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))
    return grid_search.best_estimator_

from sklearn.neighbors import KNeighborsClassifier
grid = {
    'n_neighbors': [1,2,3,4,5]
}

gs = GridSearchCV(KNeighborsClassifier(),grid)
print("Evaluate KNeighborsClassifier model")
clf = evaluate_clf(gs, features, labels, grid)


from sklearn import naive_bayes
nb_clf = naive_bayes.GaussianNB()
nb_param = {}
nb_grid_search = GridSearchCV(nb_clf, nb_param)

print("Evaluate naive bayes model")
clf = evaluate_clf(nb_grid_search, features, labels, nb_param)

"""
Evaluate KNeighborsClassifier model
accuracy: 0.900512820513
precision: 0.376
recall:    0.22069047619
n_neighbors = 5,

Evaluate naive bayes model
accuracy: 0.851794871795
precision: 0.293321428571
recall:    0.319738095238
"""

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
