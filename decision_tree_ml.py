import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Creating the DataFrame manually based on the image
data = {
    "company": [
        "google", "google", "google", "google", "google",
        "abc pharma", "abc pharma", "abc pharma", "abc pharma",
        "facebook", "facebook", "facebook", "facebook", "facebook", "facebook"
    ],
    "job": [
        "sales executive", "sales executive", "business manager", "business manager", "computer programmer",
        "sales executive", "computer programmer", "business manager", "business manager",
        "sales executive", "sales executive", "business manager", "business manager", "computer programmer", "computer programmer"
    ],
    "degree": [
        "bachelors", "masters", "bachelors", "masters", "bachelors",
        "masters", "bachelors", "bachelors", "masters",
        "bachelors", "masters", "bachelors", "masters", "bachelors", "masters"
    ],
    "salary_more_then_100k": [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Split into features (input) and target
input = df.drop("salary_more_then_100k", axis=1)
target = df['salary_more_then_100k']

# Encode categorical values
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

input['company'] = le_company.fit_transform(input['company'])
input['job'] = le_job.fit_transform(input['job'])
input['degree'] = le_degree.fit_transform(input['degree'])

# Debugging: Check if input is numerical after encoding
print("Input DataFrame after encoding:\n", input.head())




# iska matlab hai ki model.fit(input_n, target) me input_n ya target empty ho sakta hai ya valid format me nahi hai. Isliye debugging ke liye ye checks kiye the:

# input_n.shape aur target.shape print karna

# Yeh check karega ki input_n aur target empty toh nahi ho gaye.

# Agar input_n ka shape (0, 0) aaya, toh iska matlab hai ki drop() use karte waqt sab kuch delete ho gaya.

# input_n.isnull().sum() check karna

# Yeh dekhega ki kahin input_n me missing values (NaN) toh nahi hain.

# Agar missing values hain, toh scikit-learn ka DecisionTreeClassifier error de sakta hai.
# No need to drop columns again; input is already numeric
input_n = input.copy()
# input_n = input.copy()
#  kyun use kiya?
# Agar aap directly input_n = input likhoge, toh input_n sirf input ka reference ban jayega, na ki ek naya copy. Iska matlab hai ki agar aap
#  input_n me koi bhi change karoge, toh original input bhi change ho jayega.
# Debugging: Check if input_n is empty

print("Input_n shape:", input_n.shape)
print("Target shape:", target.shape)

# Check for missing values
print("Missing values in input_n:\n", input_n.isnull().sum())

# Train Decision Tree model
model = tree.DecisionTreeClassifier()
model.fit(input_n, target).score

# print("Model trained successfully!")
model.predict([[2,0,1]])
