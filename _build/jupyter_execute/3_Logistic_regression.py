#!/usr/bin/env python
# coding: utf-8

# # 3. Logistic Regression

# ## Introduction

# As the amount of available data, the strength of computing power, and the number of algorithmic improvements continue to rise, so does the importance of data science and machine learning. **Classification** techniques are an essential part of machine learning and data mining applications. Approximately 70% of problems in Data Science are classification problems. There are lots of classification problems that are available, but **logistic regression** is common and is a useful regression method for solving the binary classification problem. By the end of this tutorial, you’ll have learned about classification in general and the fundamentals of logistic regression in particular, as well as how to implement logistic regression in Python.
# 
# In this tutorial, you’ll learn:
# 
# - What logistic regression is
# - What logistic regression is used for
# - How logistic regression works
# - How to implement logistic regression in Python

# ![logit](figures/logit.PNG)

# ## What Is Classification?
# 
# **Supervised machine learning** algorithms define models that capture relationships among data. **Classification** is an area of supervised machine learning that tries to predict which class or category some entity belongs to, based on its features.
# 
# For example, you might analyze the employees of some company and try to establish a dependence on the **features** or **variables**, such as the level of education, number of years in a current position, age, salary, odds for being promoted, and so on. The set of data related to a single employee is one **observation**. The features or variables can take one of two forms:
# 
# - **Independent variables**, also called inputs or predictors, don’t depend on other features of interest (or at least you assume so for the purpose of the analysis).
# - **Dependent variables**, also called outputs or responses, depend on the independent variables.
# 
# In the above example where you’re analyzing employees, you might presume the level of education, time in a current position, and age as being mutually independent, and consider them as the inputs. The salary and the odds for promotion could be the outputs that depend on the inputs.
# 
# > Note: Supervised machine learning algorithms analyze a number of observations and try to mathematically express the dependence between the inputs and outputs. These mathematical representations of dependencies are the **models**.
# 
# The nature of the dependent variables differentiates regression and classification problems. **Regression** problems have continuous and usually unbounded outputs. An example is when you’re estimating the salary as a function of experience and education level. On the other hand, **classification problems** have discrete and finite outputs called **classes or categories**. For example, predicting if an employee is going to be promoted or not (true or false) is a classification problem.
# 
# 

# There are two main types of classification problems:
# 
# - **Binary or binomial classification**: exactly two classes to choose between (usually 0 and 1, true and false, or positive and negative)
# - **Multiclass or multinomial classification**: three or more classes of the outputs to choose from
# 
# If there’s only one input variable, then it’s usually denoted with 𝑥. For more than one input, you’ll commonly see the vector notation 𝐱 = (𝑥₁, …, 𝑥ᵣ), where 𝑟 is the number of the predictors (or independent features). The output variable is often denoted with 𝑦 and takes the values 0 or 1.

# ## When Do You Need Classification?

# You can apply classification in many fields of science and technology. For example, text classification algorithms are used to separate legitimate and spam emails, as well as positive and negative comments. You can check out Practical Text Classification With Python and Keras to get some insight into this topic. Other examples involve medical applications, biological classification, credit scoring, and more.
# 
# Image recognition tasks are often represented as classification problems. For example, you might ask if an image is depicting a human face or not, or if it’s a mouse or an elephant, or which digit from zero to nine it represents, and so on.
# 
# Logistic Regression can be used for various classification problems such as spam detection. Diabetes prediction, if a given customer will purchase a particular product or will they churn another competitor, whether the user will click on a given advertisement link or not, and many more examples are in the bucket.
# 
# Logistic Regression is one of the most simple and commonly used Machine Learning algorithms for two-class classification. It is easy to implement and can be used as the baseline for any binary classification problem. Its basic fundamental concepts are also constructive in deep learning. Logistic regression describes and estimates the relationship between one dependent binary variable and independent variables.

# ## Logistic Regression Overview

# Logistic Regression is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). In other words, the logistic regression model predicts P(Y=1) as a function of X.

# - Wiki [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
# - Video: [Introduction to Logistic Regression](https://www.youtube.com/watch?v=yhogDBEa0uQ)
# - Tutorial [Introduction to Logistic Regression](https://realpython.com/logistic-regression-python/)
# 

# ## Differences between Linear Regression and Logistic Regression

# The relation between Linear and Logistic Regression is the fact that they use labeled datasets to make predictions. However, the main difference between them is how they are being used. Linear Regression is used to solve Regression problems whereas Logistic Regression is used to solve Classification problems. 
# 
# **Classification** is about predicting a label, by identifying which category an object belongs to based on different parameters. 
# 
# **Regression** is about predicting a continuous output, by finding the correlations between dependent and independent variables.

# ![RC](figures/RC.PNG)

# ### Review of Linear Regression
# 
# Linear Regression is known as one of the simplest Machine learning algorithms that branch from Supervised Learning and is primarily used to solve regression problems. 
# 
# The use of Linear Regression is to make predictions on continuous dependent variables with the assistance and knowledge from independent variables. The overall goal of Linear Regression is to find the line of best fit, which can accurately predict the output for continuous dependent variables. Examples of continuous values are house prices, age, and salary.
# 
# Simple Linear Regression is a regression model that estimates the relationship between one single independent variable and one dependent variable using a straight line. If there are more than two independent variables, we then call this Multiple Linear Regression. 
# 
# Using the strategy of the line of best fits helps us to understand the relationship between the dependent and independent variable; which should be of linear nature. 
# 
#  
# 
# ### The Formula for Linear Regression
#  
# If you remember high school Mathematics, you will remember the formula: y = mx + b and represents the slope-intercept of a straight line. ‘y’ and ‘x’ represent variables, ‘m’ describes the slope of the line and ‘b’ describe the y-intercept, where the line crosses the y-axis. 
# 
# For Linear Regression, ‘y’ represents the dependent variable, ‘x’ represents the independent variable, 𝜷0 represents the y-intercept and 𝜷1 represents the slope, which describes the relationship between the independent variable and the dependent variable

# ![RLP](figures/RLP.jpg)

# A regression line is obtained which will give the minimum error. To do that he needs to make a line that is closest to as many points as possible.
# 
# Where ‘β1’ is the slope and ‘βo’ is the y-intercept similar to the equation of a line. The values ‘β1’ and ‘βo’ must be chosen so that they minimize the error. To check the error we have to calculate the sum of squared error and tune the parameters to try to reduce the error.

# ![cost](figures/cost.PNG)

# > Key:
# - 1. Y(predicted) is also called the hypothesis function.
# - 2. J(θ) is the cost function which can also be called the error function. Our main goal is to minimize the value of the cost.
# - 3. y(i) is the predicted output.
# - 4. hθ(x(i)) is called the hypothesis function which is basically the Y(predicted) value.

# Now the question arises, how do we reduce the error value. Well, this can be done by using Gradient Descent. The main goal of Gradient descent is to minimize the cost value. i.e. min J(θo, θ1)

# <!--![cost](figures/cost.gif) -->
# 
# ![cost](figures/cost2.PNG)

# Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do. Well, this action is analogous to calculating the gradient descent, and taking a step is analogous to one iteration of the update to the parameters.

# ![grad](figures/gradient.PNG)

# In[ ]:





# Choosing a perfect learning rate is a very important task as it depends on how large of a step we take downhill during each iteration. If we take too large of a step, we may step over the minimum. However, if we take small steps, it will require many iterations to arrive at the minimum.

# <!--![fit](figures/fit.gif)  -->
# 
# ![fit](figures/fit2.PNG) 

# ### Logistic Regression
#  
# Logistic Regression is also a very popular Machine Learning algorithm that branches off Supervised Learning. Logistic Regression is mainly used for Classification tasks. 
# 
# An example of Logistic Regression predicting whether it will rain today or not, by using 0 or 1, yes or no, or true and false. 
# 
# The use of Logistic Regression is to predict the categorical dependent variable with the assistance and knowledge of independent variables. The overall aim of Logistic Regression is to classify outputs, which can only be between 0 and 1. 
# 
# In Logistic Regression the weighted sum of inputs is passed through an activation function called Sigmoid Function which maps values between 0 and 1.
# 
# 

# ![LR](figures/LinReg.PNG)

# The Logistic Regression uses a more complex cost function, this cost function can be defined as the ‘Sigmoid function’ or also known as the ‘logistic function’ instead of a linear function.
# 
# 
# 
# The hypothesis of logistic regression tends it to limit the cost function between 0 and 1. Therefore linear functions fail to represent it as it can have a value greater than 1 or less than 0 which is not possible as per the hypothesis of logistic regression.
# 
# ![01](figures/0-1.PNG)
# 
# In order to map predicted values to probabilities, we use the Sigmoid function. The function maps any real value into another value between 0 and 1. In machine learning, we use sigmoid to map predictions to probabilities.
# 
# ![sigmoid](figures/sig.PNG)
# 
# Hypothesis Representation
# When using linear regression we used a formula of the hypothesis i.e.
# 
# hΘ(x) = β₀ + β₁X
# 
# For logistic regression we are going to modify it a little bit i.e.
# 
# σ(Z) = σ(β₀ + β₁X)
# 
# We have expected that our hypothesis will give values between 0 and 1.
# 
# Z = β₀ + β₁X
# 
# hΘ(x) = sigmoid(Z)
# 
# 
# 
# 
# 
# 

# The final hypothesis formula

# ![sigmod](figures/hp.png)

# 

# We expect our classifier to give us a set of outputs or classes based on probability when we pass the inputs through a prediction function and returns a probability score between 0 and 1.
# 
# For Example, We have 2 classes, let’s take them like cats and dogs(1 — dog , 0 — cats). We basically decide with a threshold value above which we classify values into Class 1 and of the value goes below the threshold then we classify it in Class 2.

# ![example](figures/catdog.png)

# As shown in the above graph we have chosen the threshold as 0.5, if the prediction function returned a value of 0.7 then we would classify this observation as Class 1(DOG). If our prediction returned a value of 0.2 then we would classify the observation as Class 2(CAT).

# We learnt about the cost function J(θ) in the Linear regression, the cost function represents optimization objective i.e. we create a cost function and minimize it so that we can develop an accurate model with minimum error.
# 
# The Cost Function of a Linear Regression is root mean squared error or also known as mean squared error (MSE).
# 
# ![cost](figures/cost.PNG)
# 
# MSE measures the average squared difference between an observation’s actual and predicted values. The cost will be outputted as a single number which is associated with our current set of weights. The reason we use Cost Function is to improve the accuracy of the model
# 
# If we try to use the cost function of the linear regression in ‘Logistic Regression’ then it would be of no use as it would end up being a non-convex function with many local minimums, in which it would be very difficult to minimize the cost value and find the global minimum.

# ![conv](figures/nonconv.PNG)

# For logistic regression, the Cost function is defined as:
# 
# ![conv](figures/costreg.PNG)
# 

# The above two functions can be compressed into a single function i.e.
# 
# ![conv](figures/costreg2.PNG)

# In[ ]:





# ### Gradient Descent
# Now the question arises, how do we reduce the cost value. Well, this can be done by using Gradient Descent. The main goal of Gradient descent is to minimize the cost value. i.e. min J(θ).
# 
# Now to minimize our cost function we need to run the gradient descent function on each parameter i.e.

# ![the](figures/the.PNG) 

# Gradient descent has an analogy in which we have to imagine ourselves at the top of a mountain valley and left stranded and blindfolded, our objective is to reach the bottom of the hill. Feeling the slope of the terrain around you is what everyone would do. Well, this action is analogous to calculating the gradient descent, and taking a step is analogous to one iteration of the update to the parameters.
# 
# ![grad](figures/gradient.PNG)

# - Math Tutorial for Logistic Regression [Logistic Regression](https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148)

# - Math Tutorial for Linear Regression [Linear Regression](https://towardsdatascience.com/introduction-to-linear-regression-and-polynomial-regression-f8adc96f31cb)

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

# In[ ]:





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


# ### Classification Performance
# 
# Binary classification has four possible types of results:
# 
# - True negatives: correctly predicted negatives (zeros)
# - True positives: correctly predicted positives (ones)
# - False negatives: incorrectly predicted negatives (zeros)
# - False positives: incorrectly predicted positives (ones)
# You usually evaluate the performance of your classifier by comparing the actual and predicted outputsand counting the correct and incorrect predictions.
# 
# The most straightforward indicator of classification accuracy is the ratio of the number of correct predictions to the total number of predictions (or observations). Other indicators of binary classifiers include the following:
# 
# - The positive predictive value is the ratio of the number of true positives to the sum of the numbers of true and false positives.
# - The negative predictive value is the ratio of the number of true negatives to the sum of the numbers of true and false negatives.
# - The sensitivity (also known as recall or true positive rate) is the ratio of the number of true positives to the number of actual positives.
# - The specificity (or true negative rate) is the ratio of the number of true negatives to the number of actual negatives.

# In[ ]:





# In[ ]:





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




