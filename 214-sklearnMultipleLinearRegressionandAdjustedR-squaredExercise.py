#!/usr/bin/env python
# coding: utf-8

# # Adjusted R-squared - Exercise
# 
# Using the code from the lecture, create a function which will calculate the adjusted R-squared for you, given the independent variable(s) (x) and the dependent variable (y).
# 
# Check if you function is working properly.
# 
# Please solve the exercise at the bottom of the notebook (in order to check if it is working you must run all previous cells).

# ## Import the relevant libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression


# ## Load the data

# In[2]:


data = pd.read_csv('1.02. Multiple linear regression.csv')
data.head()


# In[3]:


data.describe()


# ## Create the multiple linear regression

# ### Declare the dependent and independent variables

# In[4]:


x = data[['SAT','Rand 1,2,3']]
y = data['GPA']


# ### Regression itself

# In[5]:


reg = LinearRegression()
reg.fit(x,y)


# In[6]:


reg.coef_


# In[7]:


reg.intercept_


# ### Calculating the R-squared

# In[8]:


reg.score(x,y)


# ### Formula for Adjusted R^2
# 
# $R^2_{adj.} = 1 - (1-R^2)*\frac{n-1}{n-p-1}$

# In[9]:


x.shape


# In[10]:


r2 = reg.score(x,y)
n = x.shape[0]
p = x.shape[1]

adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
adjusted_r2


# ### Adjusted R^2 function

# In[11]:


def adjusted_r2_funtion(x,y):
    r2 = reg.score(x,y)
    n = x.shape[0]
    p = x.shape[1]
    
    adjusted_r2 = 1 - (1-r2) * (n-1)/(n-p-1)
    return adjusted_r2


# In[12]:


adjusted_r2_funtion(x,y)


# In[ ]:




