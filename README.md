# sklearn-Multiple-Linear-Regression-and-Adjusted-R-squared

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

data = pd.read_csv('1.02.Multiple_linear_regression.csv')
data.head()

data.describe()

x = data[['SAT','Rand 1,2,3']]
y = data['GPA']

reg = LinearRegression()
reg.fit(x,y)

reg.coef_

reg.intercept_

reg.score(x,y)

x.shape

r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2

# There are different ways to solve this problem
def adj_r2(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2
    
    adj_r2(x,y)
