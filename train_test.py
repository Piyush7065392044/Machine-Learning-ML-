#  Training and Testing Data
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = {
    'Mileage': [69000, 35000, 57000, 22500, 46000, 59000, 32000, 78000, 15000, 48000],
    'Age(yrs)': [6, 3, 5, 2, 4, 5, 3, 7, 1, 4],
    'Sell Price($)': [18000, 34000, 26100, 40000, 31500, 25000, 33000, 19000, 45000, 31000]
}

df = pd.DataFrame(data)
# print(df.head)
df.head()

plt.scatter(df['Mileage'], df['Sell Price($)'])
plt.show()

x = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)
len(x)
# len(x_train)
# len(x_test)
len(x)
len(x_train)
len(x_test)
x_test
x_train
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
x_test
model.predict(x_test)
model.score(x_test,y_test)