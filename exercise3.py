
# normal way to solve
# import pandas as pd

# # Creating DataFrame from the image without the removed columns
# data = {
#     "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 3],
#     "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3],
#     "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female", "female", "female", "male",
#             "male", "female", "female", "male", "male", "female", "female", "male", "male", "female", "male", "male", "female"],
#     "Age": [22, 38, 26, 35, 35, None, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, None, 31, None, 35, 34, 15, 28, 38, None],
#     "Ticket": ["A/5 21171", "PC 17599", "STON/O2.", "113803", "373450", "330877", "17463", "349909", "347742",
#                "237736", "PP 9549", "113783", "A/5. 2151", "347082", "350406", "248706", "382652", "244373",
#                "345763", "2649", "239865", "248698", "330923", "113788", "349909", "347077"],
#     "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05,
#              31.275, 7.8542, 16, 29.125, 13, 18, 7.225, 26, 13, 8.0292, 35.5, 21.075, 31.3875],
#     "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C", "S", "S", "S", "S", "S", "S", "Q", "S", "S", "C",
#                  "S", "S", "S", "S", "S", "S"]
# }

# # Converting to DataFrame
# df = pd.DataFrame(data)

# # Display the DataFrame
# print(df.head())  # Print first few rows
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()
# df['Sex'] = le.fit_transform(df['Sex'])
# df['Ticket'] = le.fit_transform(df['Ticket'])
# df['Embarked'] = le.fit_transform(df['Embarked'])
# df

# from sklearn import tree
# model = tree.DecisionTreeClassifier()

# # Make sure column names are correct (Check with df.columns)
# df.columns = df.columns.str.strip()  # Removes extra spaces if any



# # Correct column selection
# model.fit(df[['Sex', 'Ticket', 'Embarked', 'Pclass', 'Fare']], df['Survived'])
# model.predict([[1, 0,2, 71.2833, 30.5]])

# professional industry-standard approach. to solve 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Creating DataFrame
data = {
    "Survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],  # Fixed Survived issue
    "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 3],
    "Sex": ["male", "female", "female", "female", "male", "male", "male", "male", "female", "female", "female", "female",
            "male", "male", "female", "female", "male", "male", "female", "female", "male", "male", "female", "male", "male", "female"],
    "Age": [22, 38, 26, 35, 35, None, 54, 2, 27, 14, 4, 58, 20, 39, 14, 55, 2, None, 31, None, 35, 34, 15, 28, 38, None],
    "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05, 8.4583, 51.8625, 21.075, 11.1333, 30.0708, 16.7, 26.55, 8.05,
             31.275, 7.8542, 16, 29.125, 13, 18, 7.225, 26, 13, 8.0292, 35.5, 21.075, 31.3875],
    "Embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C", "S", "S", "S", "S", "S", "S", "Q", "S", "S", "C",
                 "S", "S", "S", "S", "S", "S"]
}

df = pd.DataFrame(data)

# Handling Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encoding Categorical Variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])  # Convert 'S', 'C', 'Q' into numbers

# Feature Scaling for Continuous Variables
# StandardScaler() data ko normalize karta hai taaki sabhi features same scale pe aa jayein. Yeh mean = 0 aur standard deviation = 1 bana deta hai. 🔄
scaler = StandardScaler()
df[['Fare', 'Age']] = scaler.fit_transform(df[['Fare', 'Age']])

# Selecting Features and Target
X = df[['Sex', 'Embarked', 'Pclass', 'Fare', 'Age']]
y = df['Survived']

# Splitting into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting
prediction = model.predict([[1, 2, 3, 0.5, 1.2]])  # Input must match scaled and encoded features
print(prediction)
