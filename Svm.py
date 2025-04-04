import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['flower_name'] = df['target'].apply(lambda x: iris.target_names[x])

# Filter only class 0 and class 1 data
df = df[(df['target'] == 0) | (df['target'] == 1)]

# Separate class-wise data for plotting
df0 = df[df['target'] == 0]
df1 = df[df['target'] == 1]

# Scatter plot
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], color='green', marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], color='blue', marker='+')
plt.title('Sepal Length vs Width (Class 0 vs Class 1)')
plt.show()

# Prepare data for training
x = df.drop(['target', 'flower_name'], axis='columns')
y = df['target']
#  drop() kyun use karte hain?
# Python/Pandas mein drop() function ka use hota hai DataFrame se kisi column ya row ko hataane ke liye.
# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train model
model = SVC()
model.fit(x_train, y_train)
model.score(x_test,y_test)
model.predict([[6.4,3.2,4.5,1.5]])
model_g = SVC(gamma = 10)