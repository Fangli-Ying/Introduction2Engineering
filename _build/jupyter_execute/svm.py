#!/usr/bin/env python
# coding: utf-8

# 
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akshayrb22/playing-with-data/blob/master/supervised_learning/support_vector_machine/svm.ipynb)
# 

# # Support Vector Machine Classification
# 
# 
# ## What will we do?
# 
# We will build a Support Vector Machine that will find the optimal hyperplane that maximizes the margin between two toy data classes using gradient descent.  
# 
# ![image](figures/svm.jpeg)
# 
# ## What are some use cases for SVMs?
# 
# -Classification, regression (time series prediction, etc) , outlier detection, clustering
# 
# 
# ## How does an SVM compare to other ML algorithms?
# 
# ![alt text](https://image.slidesharecdn.com/mscpresentation-140722065852-phpapp01/95/msc-presentation-bioinformatics-7-638.jpg?cb=1406012610 "Logo Title Text 1")
# 
# - As a rule of thumb, SVMs are great for relatively small data sets with fewer outliers. 
# - Other algorithms (Random forests, deep neural networks, etc.) require more data but almost always come up with very robust models.
# - The decision of which classifier to use depends on your dataset and the general complexity of the problem.
# - "Premature optimization is the root of all evil (or at least most of it) in programming." - Donald Knuth, CS Professor (Turing award speech 1974)  
# 
# 
# ## What is a Support Vector Machine?
# 
# It's a supervised machine learning algorithm which can be used for both classification or regression problems. But it's usually used for classification. Given 2 or more labeled classes of data, it acts as a discriminative classifier, formally defined by an optimal hyperplane that seperates all the classes. New examples that are then mapped into that same space can then be categorized based on on which side of the gap they fall.
# 
# ## What are Support Vectors?
# 
# ![alt text](https://www.dtreg.com/uploaded/pageimg/SvmMargin2.jpg "Logo Title Text 1")
#  
# Support vectors are the data points nearest to the hyperplane, the points of a data set that, if removed, would alter the position of the dividing hyperplane. Because of this, they can be considered the critical elements of a data set, they are what help us build our SVM. 
# 
# ## Whats a hyperplane?
# 
# ![alt text](http://slideplayer.com/slide/1579281/5/images/32/Hyperplanes+as+decision+surfaces.jpg "Logo Title Text 1")
# 
# Geometry tells us that a hyperplane is a subspace of one dimension less than its ambient space. For instance, a hyperplane of an n-dimensional space is a flat subset with dimension n − 1. By its nature, it separates the space into two half spaces.
# 
# ## Let's define our loss function (what to minimize) and our objective function (what to optimize)
# 
# #### Loss function
# 
# We'll use the Hinge loss. This is a loss function used for training classifiers. The hinge loss is used for "maximum-margin" classification, most notably for support vector machines (SVMs).
# 
# ![alt text](http://i.imgur.com/OzCwzyN.png "Logo Title Text 1")
# 
# 
# c is the loss function, x the sample, y is the true label, f(x) the predicted label.
# 
# ![alt text](http://i.imgur.com/FZ7JcG3.png "Logo Title Text 1")
# 
#  
# #### Objective Function
# 
# ![alt text](http://i.imgur.com/I5NNu44.png "Logo Title Text 1")
# 
# As you can see, our objective of a SVM consists of two terms. The first term is a regularizer, the heart of the SVM, the second term the loss. The regularizer balances between margin maximization and loss. We want to find the decision surface that is maximally far away from any data points.
# 
# How do we minimize our loss/optimize for our objective (i.e learn)?
# 
# We have to derive our objective function to get the gradients! Gradient descent ftw.  As we have two terms, we will derive them seperately using the sum rule in differentiation.
# 
# 
# ![alt text](http://i.imgur.com/6uK3BnH.png "Logo Title Text 1")
# 
# This means, if we have a misclassified sample, we update the weight vector w using the gradients of both terms, else if classified correctly,we just update w by the gradient of the regularizer.
# 
# 
# 
# Misclassification condition 
# 
# ![alt text](http://i.imgur.com/g9QLAyn.png "Logo Title Text 1")
# 
# Update rule for our weights (misclassified)
# 
# ![alt text](http://i.imgur.com/rkdPpTZ.png "Logo Title Text 1")
# 
# including the learning rate η and the regularizer λ
# The learning rate is the length of the steps the algorithm makes down the gradient on the error curve.
# - Learning rate too high? The algorithm might overshoot the optimal point.
# - Learning rate too low? Could take too long to converge. Or never converge.
# 
# The regularizer controls the trade off between the achieving a low training error and a low testing error that is the ability to generalize your classifier to unseen data. As a regulizing parameter we choose 1/epochs, so this parameter will decrease, as the number of epochs increases.
# - Regularizer too high? overfit (large testing error) 
# - Regularizer too low? underfit (large training error) 
# 
# Update rule for our weights (correctly classified)
# 
# ![alt text](http://i.imgur.com/xTKbvZ6.png "Logo Title Text 1")
# 

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

warnings.filterwarnings("ignore") 
import sklearn


# In[ ]:


url = 'https://raw.githubusercontent.com/melwinlobo18/K-Nearest-Neighbors/master/Dataset/data.csv'
df = pd.read_csv(url)  # Dataset - Breast Cancer Wisconsin Data
df['diagnosis'] = df['diagnosis'].map({
    'M': 1,
    'B': 2
})  # Label values - 1 for Malignant and 2 for Benign
labels = df['diagnosis'].tolist()
df['Class'] = labels  #Cpying values of diagnosis to newly clreated labels column
df = df.drop(['id', 'Unnamed: 32', 'diagnosis'],
             axis=1)  #Dropping unncessary columns
df.head()  #Displaying first five rows of the dataset


# In[ ]:


target_names = ['', 'M', 'B']
df['attack_type'] = df.Class.apply(lambda x: target_names[x])
df.head()


# In[ ]:


df1 = df[df.Class == 1]
df2 = df[df.Class == 2]


# In[ ]:


plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.scatter(df1['radius_mean'], df1['texture_mean'], color='green', marker='+')
plt.scatter(df2['radius_mean'], df2['texture_mean'], color='blue', marker='.')


# In[ ]:


X = df.drop(['Class', 'attack_type'], axis='columns')
X.head()


# In[ ]:


y = df.Class


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


print(len(X_train))
print(len(X_test))


# In[ ]:


model = SVC(kernel='linear')


# In[ ]:


model.fit(X_train, y_train)


# In[ ]:


predictions = model.predict(X_test)
print(predictions)


# In[ ]:


percentage = model.score(X_test, y_test)


# In[3]:


from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_test, predictions)
print("Confusion Matrix")
print(res)
print(f"Test Set: {len(X_test)}")
print(f"Accuracy = {percentage*100} %")


# <h2 style='color:blue' align="center">Support Vector Machine Tutorial Using Python Sklearn</h2>

# In[5]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# <img height=300 width=300 src="./figures/iris_petal_sepal.png" />

# In[2]:


iris.feature_names


# In[3]:


iris.target_names


# In[6]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[8]:


df['target'] = iris.target
df.head()


# In[9]:


df[df.target==1].head()


# In[10]:


df[df.target==2].head()


# In[11]:


df['flower_name'] =df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[13]:


df[45:55]


# In[15]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Sepal length vs Sepal Width (Setosa vs Versicolor)**

# In[17]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'],color="blue",marker='.')


# **Petal length vs Pepal Width (Setosa vs Versicolor)**

# In[18]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color="blue",marker='.')


# **Train Using Support Vector Machine (SVM)**

# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


X = df.drop(['target','flower_name'], axis='columns')
y = df.target


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[52]:


len(X_train)


# In[53]:


len(X_test)


# In[75]:


from sklearn.svm import SVC
model = SVC()


# In[76]:


model.fit(X_train, y_train)


# In[77]:


model.score(X_test, y_test)


# In[78]:


model.predict([[4.8,3.0,1.5,0.3]])


# **Tune parameters**

# **1. Regularization (C)**

# In[97]:


model_C = SVC(C=1)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# In[106]:


model_C = SVC(C=10)
model_C.fit(X_train, y_train)
model_C.score(X_test, y_test)


# **2. Gamma**

# In[103]:


model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
model_g.score(X_test, y_test)


# **3. Kernel**

# In[104]:


model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[105]:


model_linear_kernal.score(X_test, y_test)


# **Exercise**

# Train SVM classifier using sklearn digits dataset (i.e. from sklearn.datasets import load_digits) and then,
# 
# 1. Measure accuracy of your model using different kernels such as rbf and linear.
# 2. Tune your model further using regularization and gamma parameters and try to come up with highest accurancy score
# 3. Use 80% of samples as training data size
# 

# In[ ]:




