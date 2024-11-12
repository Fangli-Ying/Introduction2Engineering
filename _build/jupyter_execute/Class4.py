#!/usr/bin/env python
# coding: utf-8

# # Getting started in scikit-learn with the famous iris dataset
# 
#  Download the notebooks from [GitHub](https://fangli-ying.github.io).
# 
# **Note:** This notebook uses Python 3.9.1 and scikit-learn 0.23.2. 

# ## Agenda
# 
# - What is the famous iris dataset, and how does it relate to Machine Learning?
# - How do we load the iris dataset into scikit-learn?
# - How do we describe a dataset using Machine Learning terminology?
# - What are scikit-learn's four key requirements for working with data?

# ## Introducing the iris dataset

# ![Iris](img/03_iris.png)

# - 50 samples of 3 different species of iris (150 samples total)
# - Measurements: sepal length, sepal width, petal length, petal width

# In[1]:


from IPython.display import IFrame
IFrame('https://www.dataschool.io/files/iris.txt', width=300, height=200)


# ## Machine Learning on the iris dataset
# 
# - Framed as a **supervised learning** problem: Predict the species of an iris using the measurements
# - Famous dataset for Machine Learning because prediction is **easy**
# - Learn more about the iris dataset: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/Iris)

# ## Loading the iris dataset into scikit-learn

# In[2]:


# import load_iris function from datasets module
from sklearn.datasets import load_iris


# In[3]:


# save "bunch" object containing iris dataset and its attributes
iris = load_iris()
type(iris)


# In[4]:


# print the iris data
print(iris.data)


# ## Machine Learning terminology
# 
# - Each row is an **observation** (also known as: sample, example, instance, record)
# - Each column is a **feature** (also known as: predictor, attribute, independent variable, input, regressor, covariate)

# In[5]:


# print the names of the four features
print(iris.feature_names)


# In[6]:


# print integers representing the species of each observation
print(iris.target)


# In[7]:


# print the encoding scheme for species: 0 = setosa, 1 = versicolor, 2 = virginica
print(iris.target_names)


# - Each value we are predicting is the **response** (also known as: target, outcome, label, dependent variable)
# - **Classification** is supervised learning in which the response is categorical
# - **Regression** is supervised learning in which the response is ordered and continuous

# ## Requirements for working with data in scikit-learn
# 
# 1. Features and response are **separate objects**
# 2. Features should always be **numeric**, and response should be **numeric** for regression problems
# 3. Features and response should be **NumPy arrays**
# 4. Features and response should have **specific shapes**

# In[8]:


# check the types of the features and response
print(type(iris.data))
print(type(iris.target))


# In[9]:


# check the shape of the features (first dimension = number of observations, second dimensions = number of features)
print(iris.data.shape)


# In[10]:


# check the shape of the response (single dimension matching the number of observations)
print(iris.target.shape)


# In[11]:


# store feature matrix in "X"
X = iris.data

# store response vector in "y"
y = iris.target


# ## Resources
# 
# - scikit-learn documentation: [Dataset loading utilities](https://scikit-learn.org/stable/datasets.html)
# - Jake VanderPlas: Fast Numerical Computing with NumPy ([slides](https://speakerdeck.com/jakevdp/losing-your-loops-fast-numerical-computing-with-numpy-pycon-2015), [video](https://www.youtube.com/watch?v=EEUXKG97YRw))
# - Scott Shell: [An Introduction to NumPy](https://sites.engineering.ucsb.edu/~shell/che210d/numpy.pdf) (PDF)
