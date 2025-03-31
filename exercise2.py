import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Creating the DataFrame
data = {
    'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37, 0.41, 0.10, 0.92, 0.89, 0.42],
    'last_evaluation': [0.53, 0.86, 0.88, 0.87, 0.52, 0.50, 0.77, 0.85, 1.00, 0.53],
    'number_project': [2, 5, 7, 5, 2, 2, 6, 5, 5, 2],
    'average_montly_hours': [157, 262, 272, 223, 159, 153, 247, 259, 224, 142],
    'time_spend_company': [3, 6, 4, 5, 3, 3, 4, 5, 5, 3],
    'Work_accident': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'left': [1, 1, 1, 1, 1, 0, 1, 1, 0, 1],  # ✅ Added some 0 values
    'promotion_last_5years': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Department': ['sales', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales', 'sales'],
    'salary': ['low', 'low', 'low', 'low', 'low', 'low', 'low', 'low', 'low', 'low']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Label Encoding for 'salary'
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])

# Defining X and Y
x = df[['satisfaction_level', 'last_evaluation', 'number_project', 
        'average_montly_hours', 'time_spend_company', 'Work_accident', 
        'promotion_last_5years', 'salary']]
y = df['left']

# Plotting data
plt.scatter(df['satisfaction_level'], df['left'], color='blue', label='Satisfaction vs Left')
plt.xlabel('Satisfaction Level')
plt.ylabel('Left (0 or 1)')
plt.title('Satisfaction vs Employee Left')
plt.legend()
plt.show()

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

# Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# ✅ Making Predictions
predictions = model.predict(x_test)
print("Predictions:", predictions)
predictions = model.predict(x_test)
predictions