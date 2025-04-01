from re import X
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
digits
len(digits)
dir (digits)
# why use  dir(digits)
# When you run dir(digits), it lists all available attributes and methods of the digits object.

# Since digits is a Bunch object (similar to a dictionary in Scikit-Learn), dir(digits) typically returns:

# python
# Copy
# Edit
# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
digits.images[0]
for i in range (5):
  # use looop to se only frist 5 starting images 
 plt.matshow(digits.images[i])
plt.show()
plt.gray()
# plt.show()z
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# model ko import kra hai yhapar logisticregration 
from sklearn.model_selection import train_test_split
x_train ,x_test, y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2) 
# model ko split kra hai and test size =0.2 kra hai and usko test kar rhe hai 20 % data 
len(x_train)
len(x_test)
len(digits.data)
# 
len(x_train)
len(x_test)
len(digits.data)
model.fit(x_train,y_train)
model.predict(x_test)
# or yha par data predict hua hai buy prediction ok 
# 
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
array([6, 4, 0, 8, 3, 1, 5, 7, 1, 2, 7, 0, 6, 9, 7, 4, 0, 7, 1, 9, 0, 6,
       8, 1, 3, 4, 6, 6, 8, 9, 8, 2, 0, 8, 7, 5, 3, 0, 3, 9, 5, 4, 7, 6,
       4, 4, 4, 8, 2, 6, 4, 4, 6, 8, 8, 2, 5, 3, 1, 2, 5, 3, 4, 1, 7, 0,
       1, 0, 9, 4, 0, 6, 0, 2, 6, 2, 7, 9, 4, 7, 7, 7, 4, 7, 3, 8, 5, 8,
       2, 0, 8, 0, 0, 2, 0, 4, 4, 2, 7, 1, 1, 1, 0, 9, 7, 6, 1, 2, 2, 7,
       0, 9, 4, 0, 5, 7, 3, 5, 7, 3, 5, 8, 3, 8, 1, 0, 3, 1, 7, 4, 9, 2,
       4, 8, 5, 6, 4, 1, 9, 0, 9, 4, 3, 8, 0, 7, 0, 3, 9, 9, 8, 4, 3, 0,
       8, 3, 7, 4, 5, 5, 9, 5, 8, 2, 6, 6, 5, 6, 0, 1, 3, 0, 0, 2, 3, 7,
       7, 2, 6, 0, 7, 3, 4, 2, 1, 6, 6, 1, 6, 1, 5, 0, 2, 4, 4, 4, 6, 2,
       3, 8, 8, 8, 6, 7, 3, 0, 4, 3, 8, 5, 8, 6, 0, 1, 4, 2, 3, 6, 4, 3,
       1, 9, 5, 8, 1, 9, 6, 0, 4, 5, 4, 3, 6, 6, 6, 1, 9, 2, 4, 1, 9, 3,
       2, 0, 7, 4, 1, 5, 2, 3, 3, 4, 7, 6, 1, 3, 1, 7, 9, 9, 4, 4, 7, 7,
       5, 7, 6, 6, 0, 2, 4, 9, 7, 8, 5, 2, 1, 4, 9, 5, 7, 4, 8, 2, 3, 1,
       0, 6, 5, 4, 4, 6, 0, 5, 6, 2, 4, 4, 4, 3, 7, 2, 7, 7, 3, 5, 0, 3,
       5, 7, 2, 4, 9, 4, 0, 5, 7, 5, 4, 6, 8, 7, 1, 4, 1, 4, 7, 2, 1, 7,
       4, 4, 1, 2, 3, 6, 0, 8, 9, 3, 5, 6, 7, 4, 3, 9, 7, 2, 7, 3, 3, 3,
       3, 8, 8, 7, 4, 9, 5, 0])
    #    
    y_test
# ye humare pass data the phele se he 
# /
array([2, 1, 9, 9, 6, 2, 4, 7, 6, 5, 6, 8, 8, 9, 5, 7, 3, 1, 8, 6, 3, 0,
       9, 1, 0, 4, 9, 4, 5, 3, 9, 5, 9, 3, 9, 7, 1, 4, 5, 8, 2, 0, 7, 2,
       0, 7, 4, 5, 5, 9, 1, 2, 9, 4, 7, 3, 1, 1, 6, 1, 6, 5, 2, 4, 0, 2,
       9, 5, 5, 7, 0, 6, 9, 4, 5, 4, 1, 9, 0, 9, 1, 6, 4, 5, 6, 5, 3, 6,
       8, 4, 9, 0, 4, 8, 8, 3, 0, 7, 4, 0, 5, 8, 2, 2, 1, 5, 8, 3, 7, 9,
       8, 3, 2, 6, 5, 3, 0, 2, 2, 8, 5, 5, 3, 3, 8, 3, 6, 8, 9, 0, 4, 3,
       0, 9, 8, 6, 0, 1, 4, 6, 5, 6, 2, 4, 1, 6, 8, 5, 6, 0, 9, 3, 0, 2,
       5, 7, 6, 5, 8, 4, 0, 3, 9, 8, 3, 5, 8, 2, 7, 2, 7, 3, 9, 8, 5, 9,
       2, 2, 0, 3, 2, 3, 1, 5, 6, 2, 1, 2, 3, 0, 2, 3, 8, 4, 4, 5, 5, 8,
       3, 1, 4, 5, 7, 4, 8, 9, 1, 6, 2, 8, 1, 0, 1, 9, 9, 4, 2, 5, 7, 1,
       0, 5, 7, 5, 5, 1, 7, 1, 2, 0, 0, 0, 0, 6, 7, 6, 9, 8, 5, 0, 9, 2,
       9, 9, 3, 7, 2, 9, 4, 4, 7, 3, 9, 1, 4, 7, 0, 9, 9, 0, 6, 2, 2, 8,
       9, 3, 0, 4, 5, 9, 9, 3, 3, 7, 5, 8, 0, 3, 7, 4, 0, 3, 7, 8, 3, 4,
       3, 6, 8, 5, 5, 4, 2, 3, 6, 1, 5, 9, 6, 9, 6, 2, 8, 0, 9, 5, 0, 0,
       6, 6, 8, 4, 2, 7, 9, 1, 2, 7, 6, 6, 3, 1, 4, 9, 7, 1, 1, 4, 7, 2,
       4, 8, 6, 6, 7, 3, 7, 2, 8, 4, 5, 7, 7, 8, 0, 0, 5, 2, 1, 7, 6, 3,
       5, 4, 9, 1, 3, 3, 1, 2])

    #    
    model.score(x_test,y_test)
    0.9527777777777777