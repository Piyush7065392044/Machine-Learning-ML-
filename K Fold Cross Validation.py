from string import digits
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

# using logistic regration to train model 
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)

# using svm model to train 
svm = SVC()
svm.fit(X_train, y_train)
svm.score(X_test, y_test)


#  random forest  classifire 

rf = RandomForestClassifier(n_estimators=40)
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# kfold obj 
from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf

for train_index,test_index in kf.split([1,2,3,4,5,6,7,8,9]):
  print(train_index,test_index)

# to check all model step by step else use function just write function name and get result 
def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

get_score(SVC(), X_train, X_test, y_train, y_test)

# ye jo pura code hai bda sa ye pura just ak line me ho sakta hao using cross_val_score function
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)

scores_logistic = []
scores_svm = []
scores_rf = []

for train_index, test_index in folds.split(digits.data,digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], \
                                       digits.target[train_index], digits.target[test_index]
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear',multi_class='ovr'), X_train, X_test, y_train, y_test))  
    scores_svm.append(get_score(SVC(gamma='auto'), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))

# yeitna bde code ka kaam just ye ak line me ho gya 
from sklearn.model_selection import cross_val_score
cross_val_score(LogisticRegression(),digits.data,digits.target)

# svm 
cross_val_score(SVC(),digits.data,digits.target)

# random forest 
cross_val_score(RandomForestClassifier(),digits.data,digits.target)

# 
# use diffrent model and train increse cv and estimators 
cross_val_score(RandomForestClassifier(n_estimators=40),digits.data, digits.target,cv=20)
#  ye hume model kya karta hai 
#  Is output ko kya bolte hain?
# ➡️ Cross-validation scores ya validation accuracy scores
# ➡️ Ye 20 values hain, kyunki cv=20 diya tha.

# 🔢 Ye values kya represent karti hain?
# Har value ek fold ki accuracy score hai (i.e., model ne kitna sahi predict kiya us round mein).

# Format: [fold1_accuracy, fold2_accuracy, ..., fold20_accuracy]

# Range: 0.0 se 1.0 ke beech (1.0 = 100% accuracy)