# %%
import pandas as pd

# %%
# We re trying to predict whether or not a baseball player
# is inducted into the hall of fame (hof) using a classification tree

df = pd.read_csv("C:/Users/BARIS/Desktop/500hits.csv", encoding = 'latin-1')

# %%
df.head()

# %%
df.shape

# %%
df = df.drop(columns = ['PLAYER', 'CS'])

# %%
# HOF column is my target variable
# so all other features will form my inputs
# for my input matrix X include all rows and every column except column 14 
# which is the HOF  column
# remember python begins indexing from 0

X = df.iloc[:,0:13]

# %%
y = df.iloc[:,13]

# %%
# we will split up our date between a training and a testing set

from sklearn.model_selection import train_test_split


# %%
import sklearn as sk

# %%
# random_state argument shuffles the data before applying the split
# It also makes the model more reproducible

x_train, x_test, y_train, y_test = train_test_split(X,y,random_state = 17, test_size=0.2)
# 80% of data is trained
# 20% of data is tested

# %%
x_train.shape

# %%
x_test.shape

# to make sure the train test split happened correctly

# %%
from sklearn.tree import DecisionTreeClassifier

# %%
dtc = DecisionTreeClassifier()

# %%
dtc.get_params()

# %%
 # now we will fit model to this data

dtc.fit(x_train, y_train)



# %%
y_pred = dtc.predict(x_test)

# %%
# we will do a confusion matrix (error matrix) to test this model
# a confusion matrix is a matrix where entry(1,1) is true positive between actual and predicted values 
# entry (1,2) is false negative between actual (rows) and (col.s) predicted values
# entry (2,1) is false positive and
# entry (2,2) is a true negative

from sklearn.metrics import confusion_matrix as cm

print(cm(y_test, y_pred))


# %%
# 52 true positives
# 9 false negatives
# 11 false positives
# 21 true negatives
from sklearn.metrics import classification_report

# %%
print(classification_report(y_test, y_pred))

# %%
 # precision = true positive / (true pos + false positive)
# recall = true positive / (true pos + false negative)

# WHAT IS F1-SCORE ?????
# we will look at which features have had the biggest impact on our model

# %%
dtc.feature_importances_
# the importance of each feature is calculated using a measure called
# the gini importance

# %%
features = pd.DataFrame(dtc.feature_importances_, index = X.columns )

# %%
features.head(15)
# a dataframe showing importance of each feature

# %%
# it makes sense that hits are the most important
# the least important feature appears to be 3B in this case
# were going to run the model again

# %%
dtc.get_params()

# %%
# criterion argument is the function used to measure the quality of a split
# Default (criterion) = "gini" for Gini impurity
# were going to choose a criterion called "entropy"
dtc2 = DecisionTreeClassifier(criterion='entropy', ccp_alpha=0.04)

# we chose ccp_alpha = 0.04 because we think the model overfits the data

# %%
dtc2.fit(x_train,y_train)

# %%
y_pred2 = dtc2.predict(x_test)

# %%
print(cm(y_test,y_pred2))

# %%
print(classification_report(y_test,y_pred2))

# %%
# this change didn't make too much of an impact
# but we re still going to look at feature importances
# by creating a dataframe of where every element is the importance
# of a feature

features2 = pd.DataFrame(dtc2.feature_importances_, index = X.columns)

# %%
features2.head(15)

# %%
# this change is due to ccp.
# ccp removes part of the tree in order to stop overfitting


