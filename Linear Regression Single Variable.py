import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model # type: ignore

# Create a dictionary with the data
data = {
    'Year': [
        1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979,
        1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
        1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
        2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016
    ],
    'Per Capita Income (US$)': [
        3399.299037, 3768.297935, 4251.175484, 4804.463248, 5576.514583,
        5998.144346, 7062.131392, 7100.12617, 7247.967035, 7602.912681,
        8355.96812, 9434.390652, 9619.438377, 10416.53659, 10790.32872,
        11018.95585, 11482.89153, 12974.80662, 15080.28345, 16426.72548,
        16838.6732, 17266.09769, 16412.08309, 15875.58673, 15755.82027,
        16369.31725, 16699.82668, 17310.75775, 16622.67187, 17581.02414,
        18987.38241, 18601.39724, 19232.17556, 22739.42628, 25719.14715,
        29198.05569, 32738.2629, 36144.48122, 37446.48609, 32755.17682,
        38420.52289, 42334.71121, 42665.25597, 42676.46837, 41039.8936,
        35175.18898, 34229.19363
    ]
}


df = pd.DataFrame(data)
df
plt.xlabel("area",size= 20)
plt.ylabel("price",size = 20)
# plt.scatter(df['Area (sq ft)'],df['Price (INR)'])
plt.plot(df['Year'], df['Per Capita Income (US$)'], color='blue', marker='o', linestyle='-')
reg = linear_model.LinearRegression()
# reg = reg.fit(df[['Area (sq ft)']], df['Price (INR)'])
# Correctly fitting the model
reg = reg.fit(df[['Year']], df['Per Capita Income (US$)'])

reg.predict([[2090]])
# reg.predict([[3000]])