# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler

# data = {
#     "Name": [
#         "Rob", "Michael", "Mohan", "Ismail", "Kory", "Gautam", "David", "Andrea",
#         "Brad", "Angelina", "Donald", "Tom", "Arnold", "Jared", "Stark", "Ranbir",
#         "Dipika", "Priyanka", "Nick", "Alia", "Sid", "Abdul"
#     ],
#     "Age": [
#         27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29, 32,
#         40, 41, 43, 39, 41, 39
#     ],
#     "Income($)": [
#         70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000,
#         137000, 45000, 48000, 51000, 49500, 53000, 64000, 63000, 64000, 82000, 82000, 58000
#     ]
# }

# df = pd.DataFrame(data)
# # print(df)
# df.head()
# plt.scatter(df.Age,df['Income($)'])
# plt.xlabel("age",size = 30 )
# plt.ylabel("income",size = 30)

# # crating clusters and use random_state = 42
# # 🔁 Why is random_state needed?
# # KMeans starts with randomly selected centroids before it begins clustering. Because of this randomness:

# # Without random_state: Every time you run the code, you might get slightly different clusters.

# # With random_state: You get the same result every time, which helps in debugging, comparison, or sharing code.
# km = KMeans(n_clusters=3, random_state=42)
# y_predicted = km.fit_predict(df[['Age', 'Income($)']])
# y_predicted

# #
# df['cluster'] = y_predicted
# df.head()

# # creating cluster
# df['Cluster'] = y_predicted
# km.cluster_centers_
# df1 = df[df.Cluster == 0]
# df2 = df[df.Cluster == 1]
# df3 = df[df.Cluster == 2]

# plt.scatter(df[df.Cluster == 0]['Age'], df[df.Cluster == 0]['Income($)'], color='red', label='Cluster 0')
# plt.scatter(df[df.Cluster == 1]['Age'], df[df.Cluster == 1]['Income($)'], color='blue', label='Cluster 1')
# plt.scatter(df[df.Cluster == 2]['Age'], df[df.Cluster == 2]['Income($)'], color='green', label='Cluster 2')
# plt.xlabel("age",size = 30 )
# plt.ylabel("Income($)",size = 30)
# plt.legend()
# #
# scaler = MinMaxScaler()
# scaler.fit(df[['Income($)']])
# df['Income($)'] = scaler.transform(df[['Income($)']])

# scaler.fit(df[['Age']])
# df['Age'] = scaler.transform(df[['Age']])

# df.head()


# plt.scatter(df.Age,df['Income($)'])
# Import necessary libraries
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# Enable inline plotting for Jupyter Notebooks (REMOVE this line if you're not using a Jupyter Notebook)
# %matplotlib inline

# Load the CSV file (ensure the file path is correct and the file exists)

data = {
    "Name": [
        "Rob", "Michael", "Mohan", "Ismail", "Kory", "Gautam", "David", "Andrea",
        "Brad", "Angelina", "Donald", "Tom", "Arnold", "Jared", "Stark", "Ranbir",
        "Dipika", "Priyanka", "Nick", "Alia", "Sid", "Abdul"
    ],
    "Age": [
        27, 29, 29, 28, 42, 39, 41, 38, 36, 35, 37, 26, 27, 28, 29, 32,
        40, 41, 43, 39, 41, 39
    ],
    "Income($)": [
        70000, 90000, 61000, 60000, 150000, 155000, 160000, 162000, 156000, 130000,
        137000, 45000, 48000, 51000, 49500, 53000, 64000, 63000, 64000, 82000, 82000, 58000
    ]
}

df = pd.DataFrame(data)
# Display the first few rows
print(df.head())

# 🔹 Scatter plot to visualize initial data distribution
plt.scatter(df.Age, df['Income($)'])
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title("Age vs Income - Raw Data")
plt.show()

# 🔹 Step 1: Apply KMeans clustering without scaling
km = KMeans(n_clusters=3, random_state=42)  # Added random_state for reproducibility
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

# Add predicted cluster to DataFrame
df['cluster'] = y_predicted
print(df.head())

# 🔹 Step 2: Visualize clusters without scaling
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.title("Clusters without Scaling")
plt.legend()
plt.show()

# 🔹 Step 3: Preprocessing using MinMaxScaler
scaler = MinMaxScaler()

# Scale Income column
df['Income($)'] = scaler.fit_transform(df[['Income($)']])

# Scale Age column
df['Age'] = scaler.fit_transform(df[['Age']])

print(df.head())

# 🔹 Step 4: Re-apply KMeans on scaled data
km = KMeans(n_clusters=3, random_state=42)
y_predicted = km.fit_predict(df[['Age', 'Income($)']])

df['cluster'] = y_predicted

# 🔹 Step 5: Visualize new clusters after scaling
df1 = df[df.cluster == 0]
df2 = df[df.cluster == 1]
df3 = df[df.cluster == 2]

plt.scatter(df1.Age, df1['Income($)'], color='green', label='Cluster 0')
plt.scatter(df2.Age, df2['Income($)'], color='red', label='Cluster 1')
plt.scatter(df3.Age, df3['Income($)'], color='black', label='Cluster 2')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color='purple', marker='*', label='Centroid')
plt.xlabel('Age (Scaled)')
plt.ylabel('Income($) (Scaled)')
plt.title("Clusters after Scaling")
plt.legend()
plt.show()

# 🔹 Step 6: Elbow Plot to determine optimal K
sse = []  # Sum of squared errors
k_rng = range(1, 10)
for k in k_rng:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df[['Age', 'Income($)']])
    sse.append(km.inertia_)  # inertia_ is the SSE

plt.plot(k_rng, sse, marker='o')
plt.xlabel('K')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title("Elbow Method For Optimal K")
plt.grid(True)
plt.show()
