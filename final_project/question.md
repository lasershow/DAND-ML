
## 1
>Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

### Goal

Identify Enron Employees who may have committed fraud based on the public Enron financial and email dataset.

### Why ML helps with this problem

Please think that you have become an audit team to deal with this problem. The data used for this problem contains tens of thousands of mails. How long will it take to grasp this manually?

Also, do not miss the problem, can you inject a lot of resources?

Machine learning solves these problems. Machine learning will find certain regularity from large amounts of data and identify cheaters. Also, you may discover something that people do not notice.z

### outliers

I visualized the following outliers from the box plot and excluded them.

- bonus value is 0.8e+8 or more
- deferral_payments value is -0.5e+7 or less
- exercised_stock_options value is 3.0e+8 or more
- expenses value is 5000000 or more
- salary is 2.0e+7 or more
- total_payments value is 1.0e+8  or more


## 2
>What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “intelligently select features”, “properly scale features”]

### selection process

I removed unnecessary variables, using domain knowledge.For example, email does not affect fraud.

Furthermore, I selected features using RF importance.

`ex`

```py
y = df["poi"]
X = df.drop("poi",axis=1)
selector = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=7)
selector.fit(X, y)
print selector.support_
print selector.ranking_

X_selected = selector.transform(X)
print "X.shape={}, X_selected.shape={}".format(X.shape, X_selected.shape)
```

### scaling

I needed to do scaling. This is because the unit of each feature is different. Therefore, it was necessary to look up from the same scale with uniform units.

### Feature Engineering

I did `Feature Engineering` from human intuition.

`People who worked for fraud may be e-mailing more with the person who worked for fraud.`

```py
fraction_from_poi =[]
fraction_to_poi =[]
for key, row in df.iterrows():
    fraction_from_poi.append(row["from_poi_to_this_person"] / row["to_messages"])
    fraction_to_poi.append(row["from_this_person_to_poi"] / row["from_messages"])

df['fraction_from_poi'] = fraction_from_poi
df['fraction_to_poi'] = fraction_to_poi
df.head()
```

## 3
>What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

I used SVC and SGD, SVC was better than SVC itself, but SVC did not work well at the time of submission, so I chose SVC.
There is also reason that SVC confirmed the report that does not work well for this problem.

[Got a divide by zero when trying out: SVC](https://discussions.udacity.com/t/got-a-divide-by-zero-when-trying-out-svc/19823)

## 4
>What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? What parameters did you tune? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric items: “discuss parameter tuning”, “tune the algorithm”]

### What does it mean to tune the parameters of an algorithm

In machine learning, the precision varies depending on the numerical value of a parameter in a certain model. This is because we have to construct the most fitting model for the data set. Also, depending on the data set, the optimal parameters are different.

### SGD parameters

#### loss

>The ‘log’ loss gives logistic regression, a probabilistic classifier. ‘modified_huber’ is another smooth loss that brings tolerance to outliers as well as probability estimates. ‘squared_hinge’ is like hinge but is quadratically penalized. ‘perceptron’ is the linear loss used by the perceptron algorithm. The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.

- hinge
- squared_hinge


#### penalty
>The penalty (aka regularization term) to be used. Defaults to ‘l2’ which is the standard regularizer for linear SVM models. ‘l1’ and ‘elasticnet’ might bring sparsity to the model (feature selection) not achievable with ‘l2’.

Since this data set is not sparse, choose the default l2.


#### alpha : float

>Constant that multiplies the regularization term. Defaults to 0.0001 Also used to compute learning_rate when set to ‘optimal’.

- 0.0001
- 0.0002
- 0.001
- 0.01
- 0.1

## 5
>What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric items: “discuss validation”, “validation strategy”]

Validation is done by GridSearchCV. Validation is mandatory to see if you are over learning.\

## 6
>Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

In this survey we want to prevent classifying fraud workers as not working wrong. This is to allow for preliminary use of the classification system. Therefore, recall is used for the evaluation function
