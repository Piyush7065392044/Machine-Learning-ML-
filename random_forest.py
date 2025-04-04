# RANDOM FOREST 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
# checking what is digits 
digits
digits.data
df=pd.DataFrame(digits.data)
df.head()

# graph
plt.gray()
for i in range(5):
  plt.matshow(digits.images[i])
dir(digits)
digits.target_names

dir(digits)
digits.target[:5]
df = pd.DataFrame(digits.data)
df.head()
df['target'] = digits.target
df.head()

x = df.drop('target', axis='columns')
x
y = df.target
y
# imp
# ⚠️ Agar drop na karein toh kya hoga?
# Agar tum drop('target') na karo aur saare columns ko x me de do, toh:

# Tumhara model apne aap hi answer (target) bhi input me dekh lega 😅

# Iska matlab — model cheating karega aur accuracy 100% aayegi

# But test pe fail ho jaayega (Overfitting)
# 
# 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

len(x_train)
len(y_train)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators= 124)  # Fixed: Added parentheses to initialize the model
model.fit(x_train, y_train)
model.score(x_test,y_test)
# Target Kyu Late Hain?
# Jab hum Machine Learning (ML) model train karte hain, toh usse kuch seekhne ke liye ek goal (target) chahiye hota hai
# . Target basically woh cheez hoti hai jo model ko predict karni hoti hai.

# Agar target na ho, toh model ko pata hi nahi chalega ki correct answer kya hona chahiye,
#  aur wo sahi tarike se train nahi ho paayega.
from sklearn.metrics import confusion_matrix
y_predicted = model.predict(x_test)
cm = confusion_matrix(y_test, y_predicted)


