# Regression Multiple Variables

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# Creating the dataset
data = {
    'area': [2600, 3000, 3200, 3600, 4000],
    'bedrooms': [3.0, 4.0, None, 3.0, 5.0],
    'age': [20, 15, 18, 30, 8],
    'price': [550000, 565000, 610000, 595000, 760000]
}
df = pd.DataFrame(data)
df

plt.scatter(df['area'],  df['price'],marker='*')


import math
# to use this we can replace the value and change the value jo null ho 
model_bedroom = math.floor(df.bedrooms.median())
# median_bedroom= model_bedroom
df.bedrooms = df.bedrooms.fillna(median_bedroom)
df.bedrooms
df
reg = LinearRegression()
reg = reg.fit(df[['area','bedrooms','age']],df.price)
reg.predict([[3000,3,40]])




#  second method to solve 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = {
    'experience': ['eight', 'eight', 'five', 'two', 'seven', 'three', 'ten', 'eleven'],
    'test_score(out of 10)': [8, 8, 6, 10, 9, 7, 7, 7],
    'interview_score(out of 10)': [9, 6, 7, 10, 6, 10, 7, 8],
    'salary($)': [50000, 45000, 60000, 65000, 70000, 62000, 72000, 80000]
}

df = pd.DataFrame(data)

# # Encode 'experience' using LabelEncoder
le = LabelEncoder()
df['experience'] = le.fit_transform(df['experience'])

# Scatter plot (optional, uncomment if needed)
# plt.scatter(df['experience'], df['salary($)'], color='green', marker='*')
# plt.xlabel('experience', size=20)
# plt.ylabel('salary', size=20)
# plt.show()

# Create and train the model
reg = LinearRegression()
reg.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

# Corrected prediction call
prediction = reg.predict([[2, 2, 2]])
# print("Predicted Salary:", prediction[0])


#  Why use LabelEncoder for 'experience'?
# LabelEncoder is used to convert categorical data into numerical values so that machine learning models can process the data effectively.

# ✅ When to Use LabelEncoder
# If your experience column contains categorical (non-numeric) values such as:

# plaintext
# Copy
# Edit
# zero, one, two, three, five, seven, ten, eleven
# LabelEncoder converts these labels into numbers:

# plaintext
# Copy
# Edit
# zero → 0, one → 1, two → 2, ..., eleven → 7
# ⚡️ Why Not Use It in Your Case?
# In your code, you used:

# python
# Copy
# Edit
# d.experience = d.experience.apply(w2n.word_to_num)
# w2n.word_to_num already converts words like "zero," "five," and "seven" into numeric values.

# So, LabelEncoder is not necessary here.

# 🚨 When to Use LabelEncoder Instead:
# When the column contains string labels that are not easily convertible using w2n, like:

# plaintext
# Copy
# Edit
# Beginner, Intermediate, Expert
# You would use:

# python
# Copy
# Edit
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df['experience'] = le.fit_transform(df['experience'])
# Converts:

# plaintext
# Copy
# Edit
# Beginner → 0, Intermediate → 1, Expert → 2
# ⚠️ Avoid Using LabelEncoder When:
# When data is ordinal or numeric like zero, five, seven, etc.

# w2n.word_to_num is more appropriate in such cases, which you are already using.

# So, no need for LabelEncoder in your code since you’re correctly using w2n.word_to_num! 🎯




# 


# 🤔 Why Use LabelEncoder for 'experience'?

# ✅ LabelEncoder converts text into numbers so models can process it. 
# For example:
# - zero, one, two → 0, 1, 2.

# ⚡️ Why Not Needed Here?
# You used:
# d.experience = d.experience.apply(w2n.word_to_num)
# This already converts words to numbers, so LabelEncoder is unnecessary.

# 🚨 When to Use LabelEncoder?
# If the data has labels like:
# - Beginner, Intermediate, Expert
# Use:
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df['experience'] = le.fit_transform(df['experience'])
# This converts:
# - Beginner → 0, Intermediate → 1, Expert → 2

# ⚠️ Conclusion:
# Since w2n.word_to_num works perfectly, LabelEncoder is **not required**. 👍

# 