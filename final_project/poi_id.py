#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Machine Learning
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import svm, datasets, cross_validation
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi','bonus','deferred_income','exercised_stock_options','expenses',
                 'salary','total_payments','fraction_from_poi']



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame(data_dict).T

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
df = df[df.bonus < 0.8e+8 ]
df = df[df.deferred_income >  -0.5e+7 ]
df = df[df.exercised_stock_options < 3.0e+8 ]
df = df[df.expenses < 5000000 ]
df = df[df.salary < 2.0e+7 ]
df = df[df.total_payments < 1.0e+8 ]
### Task 3: Create new feature(s)
fraction_from_poi =[]
fraction_to_poi =[]
for key, row in df.iterrows():
    fraction_from_poi.append(row["from_poi_to_this_person"] / row["to_messages"])
    fraction_to_poi.append(row["from_this_person_to_poi"] / row["from_messages"])

df['fraction_from_poi'] = fraction_from_poi
df['fraction_to_poi'] = fraction_to_poi
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

grid = {
    'alpha': [0.0001,0.0002,0.001,0.01,0.1], # learning rate
    'max_iter': [1000], # number of epochs
    'loss': ['hinge' , 'squared_hinge'],
    'penalty': ['l2'],
    'n_jobs': [-1]
}

score = 'recall'
gs = GridSearchCV(
    linear_model.SGDClassifier(),
    grid,
    cv=5,
    scoring='%s_weighted' % score )

gs.fit(X,y)

# Pass the best algorithm
clf = gs.best_estimator_


print("# Tuning hyper-parameters for %s" % score)
print()
print("Best parameters set found on development set: %s" % gs.best_params_)
print()
print("Grid scores on development set:")
print()
for params, mean_score, scores in gs.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
