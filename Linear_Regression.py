#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("/Users/tugcesandikli/Downloads/linear_regression_dataset.csv")


# In[3]:


df


# In[6]:


df = pd.read_csv("/Users/tugcesandikli/Downloads/linear_regression_dataset.csv", sep=";")


# In[7]:


df.head(10)


# In[8]:


plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


# y = b0+ b1*x

# In[13]:


x = df.deneyim.values
x


# In[28]:


x.shape


# In[29]:


x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)


# In[30]:


from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()


# In[31]:


linear_reg.fit(x,y)


# In[33]:


b0 = linear_reg.predict([[0]])
print("b0",b0)


# In[34]:


bo_ = linear_reg.intercept_
bo_


# In[35]:


b1 = linear_reg.coef_


# In[36]:


b1


# y = b0+ b1*x

# In[37]:


new_salary = 1663 + 1138 * 11


# In[39]:


print("11 yıllık deneyim olan birinin maasi: ",new_salary)


# In[40]:


b11 = linear_reg.predict([[11]])
b11


# In[41]:


y_head = linear_reg.predict(x)


# In[42]:


plt.plot(x,y_head,color="red")
plt.scatter(x,y)
plt.show()


# In[43]:


mse = np.square(np.subtract(y,y_head)).mean()


# In[44]:


print("Mean Squarred Error:",mse)


# In[45]:


from sklearn.metrics import r2_score
print("R^2 Square:",r2_score(y,y_head))


# In[ ]:




