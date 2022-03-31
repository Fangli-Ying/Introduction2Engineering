#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# ## Environment setup

# In[1]:


import platform

print(f"Python version: {platform.python_version()}")
assert platform.python_version_tuple() >= ("3", "6")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


# In[2]:


# Setup plots
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = 10, 8
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sns.set()


# In[3]:


import sklearn

print(f"scikit-learn version: {sklearn.__version__}")
assert sklearn.__version__ >= "0.20"

from sklearn.datasets import make_classification, make_blobs
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import classification_report


# In[4]:


def plot_data(x, y):
    """Plot some 2D data"""

    fig, ax = plt.subplots()
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower right", title="Classes")
    ax.add_artist(legend1)
    plt.xlim((min(x[:, 0]) - 0.1, max(x[:, 0]) + 0.1))
    plt.ylim((min(x[:, 1]) - 0.1, max(x[:, 1]) + 0.1))


def plot_decision_boundary(pred_func, x, y, figure=None):
    """Plot a decision boundary"""

    if figure is None:  # If no figure is given, create a new one
        plt.figure()
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - 0.5, x[:, 0].max() + 0.5
    y_min, y_max = x[:, 1].min() - 0.5, x[:, 1].max() + 0.5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    cm_bright = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu, alpha=0.8)


# ## Binary classification

# ### Problem formulation
# 
# Logistic regression is a classification algorithm used to estimate the probability that a data sample belongs to a particular class.
# 
# A logistic regression model computes a weighted sum of the input features (plus a bias term), then applies the [logistic function](activations:sigmoid) to this sum in order to output a probability.
# 
# $$y' = \mathcal{h}_\theta(\pmb{x}) = \sigma(\pmb{\theta}^T\pmb{x})$$
# 
# The function output is thresholded to form the model's prediction:
# 
# - $0$ if $y' \lt 0.5$
# - $1$ if $y' \geqslant 0.5$

# ### Loss function: Binary Crossentropy (log loss)
# 
# See [loss definition](loss:bce) for details.

# ### Model training
# 
# - No analytical solution because of the non-linear $\sigma()$ function: gradient descent is the only option.
# - Since the loss function is convex, GD (with the right hyperparameters) is guaranteed to find the global loss minimum.
# - Different GD optimizers exist: *newton-cg*, *l-bfgs*, *sag*... *Stochastic gradient descent* is another possibility, efficient for large numbers of samples and features.
# 
# $$\nabla_{\theta}\mathcal{L}(\pmb{\theta}) = \begin{pmatrix}
#        \ \frac{\partial}{\partial \theta_0} \mathcal{L}(\boldsymbol{\theta}) \\
#        \ \frac{\partial}{\partial \theta_1} \mathcal{L}(\boldsymbol{\theta}) \\
#        \ \vdots \\
#        \ \frac{\partial}{\partial \theta_n} \mathcal{L}(\boldsymbol{\theta})
#      \end{pmatrix} = \frac{2}{m}\pmb{X}^T\left(\sigma(\pmb{X}\pmb{\theta}) - \pmb{y}\right)$$

# ### Example: classify planar data

# In[5]:


# Generate 2 classes of linearly separable data
x_train, y_train = make_classification(
    n_samples=1000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=26,
    n_clusters_per_class=1,
)
plot_data(x_train, y_train)


# In[6]:


# Create a Logistic Regression model based on stochastic gradient descent
# Alternative: using the LogisticRegression class which implements many GD optimizers
lr_model = SGDClassifier(loss="log")

# Train the model
lr_model.fit(x_train, y_train)

print(f"Model weights: {lr_model.coef_}, bias: {lr_model.intercept_}")


# In[7]:


# Print report with classification metrics
print(classification_report(y_train, lr_model.predict(x_train)))


# In[8]:


# Plot decision boundary
plot_decision_boundary(lambda x: lr_model.predict(x), x_train, y_train)


# ## Multivariate regression

# ### Problem formulation
# 
# **Multivariate regression**, also called *softmax regression*, is a generalization of logistic regression for multiclass classification.
# 
# A softmax regression model computes the scores $s_k(\pmb{x})$ for each class $k$, then estimates probabilities for each class by applying the [softmax](activations:softmax) function to compute a probability distribution.
# 
# For a sample $\pmb{x}^{(i)}$, the model predicts the class $k$ that has the highest probability.
# 
# $$s_k(\pmb{x}) = {\pmb{\theta}^{(k)}}^T\pmb{x}$$
# 
# $$\mathrm{prediction} = \underset{k}{\mathrm{argmax}}\;\sigma(s(\pmb{x}^{(i)}))_k$$
# 
# Each class $k$ has its own parameter vector $\pmb{\theta}^{(k)}$.

# ### Model output
# 
# - $\pmb{y}^{(i)}$ (*ground truth*): **binary vector** of $K$ values. $y^{(i)}_k$ is equal to 1 if the $i$th sample's class corresponds to $k$, 0 otherwise.
# - $\pmb{y}'^{(i)}$: **probability vector** of $K$ values, computed by the model. $y'^{(i)}_k$ represents the probability that the $i$th sample belongs to class $k$.
# 
# $$\pmb{y}^{(i)} = \begin{pmatrix}
#        \ y^{(i)}_1 \\
#        \ y^{(i)}_2 \\
#        \ \vdots \\
#        \ y^{(i)}_K
#      \end{pmatrix} \in \pmb{R}^K\;\;\;\;
# \pmb{y}'^{(i)} = \begin{pmatrix}
#        \ y'^{(i)}_1 \\
#        \ y'^{(i)}_2 \\
#        \ \vdots \\
#        \ y'^{(i)}_K
#      \end{pmatrix} = \begin{pmatrix}
#        \ \sigma(s(\pmb{x}^{(i)}))_1 \\
#        \ \sigma(s(\pmb{x}^{(i)}))_2 \\
#        \ \vdots \\
#        \ \sigma(s(\pmb{x}^{(i)}))_K
#      \end{pmatrix} \in \pmb{R}^K$$

# ### Loss function: Categorical Crossentropy
# 
# See [loss definition](loss:cce) for details.

# ### Model training
# 
# Via gradient descent:
# 
# $$\nabla_{\theta^{(k)}}\mathcal{L}(\pmb{\theta}) = \frac{1}{m}\sum_{i=1}^m \left(y'^{(i)}_k - y^{(i)}_k \right)\pmb{x}^{(i)}$$
# 
# $$\pmb{\theta}^{(k)}_{next} = \pmb{\theta}^{(k)} - \eta\nabla_{\theta^{(k)}}\mathcal{L}(\pmb{\theta})$$

# ### Example: classify multiclass planar data

# In[9]:


# Generate 3 classes of linearly separable data
x_train_multi, y_train_multi = make_blobs(n_samples=1000, n_features=2, centers=3, random_state=11)

plot_data(x_train_multi, y_train_multi)


# In[10]:


# Create a Logistic Regression model based on stochastic gradient descent
# Alternative: using LogisticRegression(multi_class="multinomial") which implements SR
lr_model_multi = SGDClassifier(loss="log")

# Train the model
lr_model_multi.fit(x_train_multi, y_train_multi)

print(f"Model weights: {lr_model_multi.coef_}, bias: {lr_model_multi.intercept_}")


# In[11]:


# Print report with classification metrics
print(classification_report(y_train_multi, lr_model_multi.predict(x_train_multi)))


# In[12]:


# Plot decision boundaries
plot_decision_boundary(lambda x: lr_model_multi.predict(x), x_train_multi, y_train_multi)


# <h1>Titanic Disaster Survival Using Logistic Regression</h1>

# In[1]:


#import libraries


# In[27]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# **Load the Data**

# In[2]:


#load data


# In[28]:


titanic_data=pd.read_csv('titanic_train.csv')


# In[29]:


len(titanic_data)


# **View the data using head function which returns top  rows**

# In[30]:


titanic_data.head()


# In[31]:


titanic_data.index


# In[32]:


titanic_data.columns


# In[34]:


titanic_data.info()


# In[35]:


titanic_data.dtypes


# In[36]:


titanic_data.describe()


# **Explaining Dataset**
# 
# survival : Survival 0 = No, 1 = Yes <br>
# pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd <br>
# sex : Sex <br>
# Age : Age in years <br>
# sibsp : Number of siblings / spouses aboard the Titanic 
# <br>parch # of parents / children aboard the Titanic <br>
# ticket : Ticket number fare Passenger fare cabin Cabin number <br>
# embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton <br>
# 
# 
# 

# Data Analysis

# Import Seaborn for visually analysing the data， Find out how many survived vs Died using countplot method of seaboarn

# In[3]:


#countplot of subrvived vs not  survived


# In[37]:


sns.countplot(x='Survived',data=titanic_data)


# **Male vs Female Survival**

# In[4]:


#Male vs Female Survived?


# In[38]:


sns.countplot(x='Survived',data=titanic_data,hue='Sex')


# **See age group of passengeres travelled **<br>
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph

# In[5]:


#Check for null


# In[39]:


titanic_data.isna()


# In[6]:


#Check how many values are null


# In[40]:


titanic_data.isna().sum()


# In[7]:


#Visualize null values


# In[41]:


sns.heatmap(titanic_data.isna())


# In[8]:


#find the % of null values in age column


# In[46]:


(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100


# In[9]:


#find the % of null values in cabin column


# In[47]:


(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100


# In[10]:


#find the distribution for the age column


# In[48]:


sns.displot(x='Age',data=titanic_data)


# Data Cleaning

# **Fill the missing values**<br> we will fill the missing values for age. In order to fill missing values we use fillna method.<br> For now we will fill the missing age by taking average of all age 

# In[11]:


#fill age column


# In[51]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# **We can verify that no more null data exist** <br> we will examine data by isnull mehtod which will return nothing

# In[12]:


#verify null value


# In[52]:


titanic_data['Age'].isna().sum()


# **Alternatively we will visualise the null value using heatmap**<br>
# we will use heatmap method by passing only records which are null. 

# In[13]:


#visualize null values


# In[54]:


sns.heatmap(titanic_data.isna())


# In[ ]:





# **We can see cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it**

# In[14]:


#Drop cabin column


# In[55]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[15]:


#see the contents of the data


# In[56]:


titanic_data.head()


# **Preaparing Data for Model**<br>
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[16]:


#Check for the non-numeric column


# In[57]:


titanic_data.info()


# In[58]:


titanic_data.dtypes


# **We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems Name,Embarked and Ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Sex Column to dummies numerical values******

# In[17]:


#convert sex column to numerical values


# In[61]:


gender=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[62]:


titanic_data['Gender']=gender


# In[64]:


titanic_data.head()


# In[18]:


#drop the columns which are not required


# In[65]:


titanic_data.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)


# In[66]:


titanic_data.head()


# In[19]:


#Seperate Dependent and Independent variables


# In[67]:


x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']


# In[69]:


y


# Data Modelling

# **Building Model using Logestic Regression**

# **Build the model**

# In[20]:


#import train test split method


# In[70]:


from sklearn.model_selection import train_test_split


# In[21]:


#train test split


# In[71]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[22]:


#import Logistic  Regression


# In[72]:


from sklearn.linear_model import LogisticRegression


# In[23]:


#Fit  Logistic Regression 


# In[73]:


lr=LogisticRegression()


# In[74]:


lr.fit(x_train,y_train)


# In[24]:


#predict


# In[75]:


predict=lr.predict(x_test)


# Testing

# **See how our model is performing**

# In[25]:


#print confusion matrix 


# In[76]:


from sklearn.metrics import confusion_matrix


# In[79]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'],index=['Actual No','Actual Yes'])


# 

# In[26]:


#import classification report


# In[81]:


from sklearn.metrics import classification_report


# In[82]:


print(classification_report(y_test,predict))


# **Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or  by using other model**
# 
# Note: <br>
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations <br>
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class
# F1 score - F1 Score is the weighted average of Precision and Recall.
# 
# 

# In[ ]:




