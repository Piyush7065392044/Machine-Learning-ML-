import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Create data dictionary
data = {
    'Car Model': [
        'BMW X5', 'BMW X5', 'BMW X5', 'BMW X5', 'BMW X5',
        'Audi A5', 'Audi A5', 'Audi A5', 'Audi A5',
        'Mercedez Benz C class', 'Mercedez Benz C class',
        'Mercedez Benz C class', 'Mercedez Benz C class'
    ],
    'Mileage': [69000, 35000, 57000, 22500, 46000,
                59000, 52000, 72000, 91000,
                67000, 83000, 79000, 59000],
    'Sell Price($)': [18000, 34000, 26100, 40000, 31500,
                      29400, 32000, 19300, 12000,
                      22000, 20000, 21000, 33000],
    'Age(yrs)': [6, 3, 5, 2, 4, 5, 5, 6, 8, 6, 7, 7, 5]
}

# Create DataFrame
df = pd.DataFrame(data)
print(df)

# One-hot encoding for 'Car Model'
dummies = pd.get_dummies(df['Car Model'])
df_dummies = pd.concat([df, dummies], axis='columns')
df_dummies.drop(['Car Model'], axis='columns', inplace=True)

# Define X and y
x = df_dummies.drop('Sell Price($)', axis='columns')
y = df_dummies['Sell Price($)']

# Train model
model = LinearRegression()
model.fit(x, y)

# Predict with correct input
predicted_price = model.predict([[4500, 5, 0, 1, 0]])
print("Predicted Price: ", predicted_price[0])

# Plot Mileage vs Sell Price
plt.plot(df['Mileage'], df['Sell Price($)'], color='blue', marker='o', linestyle='-')
plt.xlabel('Mileage')
plt.ylabel('Sell Price ($)')
plt.title('Mileage vs Sell Price Plot')
plt.show()
