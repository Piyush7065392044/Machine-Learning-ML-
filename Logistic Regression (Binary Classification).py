import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 18, 28, 27, 29, 49, 55, 25, 58, 19, 18, 21, 26, 45, 40, 54, 23],
    'bought_insurance': [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
df.head()

plt.scatter(df.age,df.bought_insurance)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)
len(x_train)
len(x_test)
len(y_train)
len(y_test)
x_test

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
x_test
model.predict(x_test)
model.predict_log_proba([[47]])
# probality cheack karta hai probality fucntion ki kitna % chhance hai insorance lane ki  
model.predict_proba([[1]])
model.score(x_test,y_test)