# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# 1. Create dataset with town, area, and price
data = {
    'town': ['monroe township', 'monroe township', 'monroe township', 'monroe township', 'monroe township',
             'west windsor', 'west windsor', 'west windsor', 'west windsor',
             'robinsville', 'robinsville', 'robinsville', 'robinsville'],
    'area': [2600, 3000, 3200, 3600, 4000, 2600, 2800, 3300, 3600, 2600, 2900, 3100, 3600],
    'price': [550000, 565000, 610000, 680000, 725000, 585000, 615000, 650000, 710000, 575000, 600000, 620000, 695000]
}

# 2. Convert data into DataFrame
df = pd.DataFrame(data)

# 3. Create dummy variables for 'town' (one-hot encoding)
dummies = pd.get_dummies(df.town)

# 4. Combine original data with dummy columns
df_dummies = pd.concat([df, dummies], axis='columns')

# 5. Drop original 'town' and one dummy column to avoid multicollinearity
df_dummies.drop(['town', 'west windsor'], axis='columns', inplace=True)

# 6. Define X (features) and Y (target)
x = df_dummies.drop('price', axis='columns')  # area, monroe township, robinsville
y = df_dummies.price                          # price

# 7. Train the Linear Regression model
model = LinearRegression()
model.fit(x, y)

# 8. Predict price for different conditions
print("Prediction 1:", model.predict([[3400, 0, 0]])[0])  # west windsor
print("Prediction 2:", model.predict([[3400, 1, 0]])[0])  # monroe township
print("Prediction 3:", model.predict([[3400, 0, 1]])[0])  # robinsville

# 9. Plot Area vs Price graph
plt.plot(df['area'], df['price'], color='blue', marker='o', linestyle='-')
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($)')
plt.title('Area vs Price Plot')
plt.show()

# 10. Using LabelEncoder to convert 'town' to numerical values
le = LabelEncoder()
df['town'] = le.fit_transform(df['town'])  # Encodes: monroe township -> 2, west windsor -> 1, robinsville -> 0

# 11. Define X and Y using LabelEncoder
x = df[['area', 'town']].values  # area and encoded town
y = df['price'].values

# 12. Train the model again with encoded values
model.fit(x, y)

# 13. Predict price for town 1 (west windsor) and area 2600
predicted_price = model.predict([[2600, 1]])[0]
print("Your data is " + str(predicted_price))

# 14. Model accuracy (R^2 score)
accuracy = model.score(x, y)
print(f"Model accuracy (R^2 score): {accuracy * 100:.2f}%")






# 
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression


# data = {
#     'town': ['monroe township', 'monroe township', 'monroe township', 'monroe township', 'monroe township',
#              'west windsor', 'west windsor', 'west windsor', 'west windsor',
#              'robinsville', 'robinsville', 'robinsville', 'robinsville'],
#     'area': [2600, 3000, 3200, 3600, 4000, 2600, 2800, 3300, 3600, 2600, 2900, 3100, 3600],
#     'price': [550000, 565000, 610000, 680000, 725000, 585000, 615000, 650000, 710000, 575000, 600000, 620000, 695000]
# }

# df = pd.DataFrame(data)
# df

# dummies = pd.get_dummies(df.town)
# dummies

# df_dummies = pd.concat([df, dummies], axis='columns')

# df_dummies

# df_dummies.drop(['town','west windsor'],axis='columns',inplace=True)
# df_dummies

# x = df_dummies.drop('price',axis='columns')
# x 

# y = df_dummies.price
# y 

# model = LinearRegression()
# model.fit(x,y)

# model.predict([[3400,0,0]])

# model.predict([[3400,1,0]])

# model.predict([[3400,1,0]])
# plt.plot(df['area'], df['price'], color='blue', marker='o', linestyle='-')


# dfile = df 
# dfile 

# # conver english elphabets in number using [python tranform function that covert value into 0 and 1 ]
# from sklearn.preprocessing import LabelEncoder

# # Creating a LabelEncoder object
# le = LabelEncoder()

# # Encoding 'town' column with numerical values
# dfile['town'] = le.fit_transform(dfile['town'])
# dfile  # Displaying the modified DataFrame


# x = dfile[['area','town']].values
# x
# y = dfile[['price']].values
# y


# model = LinearRegression()
# model.fit(x,y)

# # Correct way to print the result
# predicted_price = model.predict([[1, 2600]])  # Get the predicted value
# print("Your data is " + str(predicted_price))

# model.score(x,y)


# for accuracy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Data dictionary
data = {
    'town': ['monroe township', 'monroe township', 'monroe township', 'monroe township', 'monroe township',
             'west windsor', 'west windsor', 'west windsor', 'west windsor',
             'robinsville', 'robinsville', 'robinsville', 'robinsville'],
    'area': [2600, 3000, 3200, 3600, 4000, 2600, 2800, 3300, 3600, 2600, 2900, 3100, 3600],
    'price': [550000, 565000, 610000, 680000, 725000, 585000, 615000, 650000, 710000, 575000, 600000, 620000, 695000]
}

# Creating DataFrame
df = pd.DataFrame(data)
df

# Using LabelEncoder to transform 'town' into numerical values
le = LabelEncoder()
df['town'] = le.fit_transform(df['town'])
df  # Transformed DataFrame

# Defining X (features) and Y (target)
x = df[['town', 'area']].values  # Features: town and area
y = df['price'].values           # Target: price

# Creating and training the LinearRegression model
model = LinearRegression()
model.fit(x, y)

# Model prediction (example prediction for town 1, area 2600)
predicted_price = model.predict([[1, 2600]])[0]
print("Your data is " + str(predicted_price))

# Calculating model accuracy
accuracy = model.score(x, y)
print(f"Model accuracy (R^2 score): {accuracy * 100:.2f}%")

# Plotting the data
plt.plot(df['area'], df['price'], color='blue', marker='o', linestyle='-')
plt.xlabel('Area (sqft)')
plt.ylabel('Price ($)')
plt.title('Area vs Price Plot')
plt.show()
