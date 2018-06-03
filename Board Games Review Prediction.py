
# coding: utf-8

# In[2]:


import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print(pandas.__version__)
print(matplotlib.__version__)
print(seaborn.__version__)
print(sklearn.__version__)


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[4]:


#Load the data
games = pandas.read_csv("games.csv")


# In[5]:


#print the names of the columns in games
print(games.columns)

print(games.shape)


# In[8]:


# Make histogram of all the ratings and average_rating column
plt.hist(games["average_rating"])
plt.show()


# In[9]:


#print the first row of all the games with zero scores
print(games[games["average_rating"] == 0].iloc[0])

#print the first row of games with scores greater than zero
print(games[games["average_rating"] > 0].iloc[0])


# In[12]:


#remove any rows without user reviews
games = games[games["users_rated"] > 0]

#remove any rows with missing values
games = games.dropna(axis=0)

#make a histogram of all the average ratings
plt.hist(games["average_rating"])
plt.show()


# In[14]:


print(games.columns)


# In[16]:


#correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12,9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# In[17]:


# get all the columns from the dataframe
columns = games.columns.tolist()

#filter the columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

#store the variable we'll be predicting on
target = "average_rating"


# In[18]:


#generate training and test datasets
from sklearn.model_selection import train_test_split

#generate training set
train = games.sample(frac=0.8, random_state = 1)

#select anything not in the training set and put it in test
test = games.loc[~games.index.isin(train.index)]

#print shapes
print(train.shape)
print(test.shape)


# In[20]:


# import linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#initialize the model class
LR = LinearRegression()

#fit the model the training data
LR.fit(train[columns], train[target])


# In[21]:


# generate prediction for the test set
predictions = LR.predict(test[columns])

#compute error between our test predictions and actual values
mean_squared_error(predictions, test[target])


# In[22]:


#import the random forest model
from sklearn.ensemble import RandomForestRegressor

#initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf=10, random_state = 1)

#fit to the data
RFR.fit(train[columns], train[target])


# In[23]:


#make predictions
predictions = RFR.predict(test[columns])

#compute the error between our test predictions and actual values
mean_squared_error(predictions, test[target])


# In[27]:


test[columns].iloc[0]


# In[28]:


#make prediction with both models
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

#print out the predictions
print(rating_LR)
print(rating_RFR)


# In[30]:


test[target].iloc[0] 

