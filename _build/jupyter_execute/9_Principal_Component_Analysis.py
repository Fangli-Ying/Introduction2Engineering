#!/usr/bin/env python
# coding: utf-8

# # 9. Principal Component Analysis

# Up until now, we have been looking in depth at supervised learning estimators: those estimators that predict labels based on labeled training data. Here we begin looking at several unsupervised estimators, which can highlight interesting aspects of the data without reference to any known labels.

# When implementing machine learning algorithms, the inclusion of more features might lead to worsening performance issues. Increasing the number of features will not always improve classification accuracy, which is also known as the curse of dimensionality. Hence, we apply dimensionality reduction to improve classification accuracy by selecting the optimal set of lower dimensionality features.
# 
# **Principal component analysis (PCA)** is essential for data science, machine learning, data visualization, statistics, and other quantitative fields.

# ![Dim](figures/dim.png)

# There are two techniques to make dimensionality reduction:
# 
# - Feature Selection
# - Feature Extraction
# 
# It is essential to know about vector, matrix, and transpose matrix, eigenvalues, eigenvectors, and others to understand the concept of dimensionality reduction.

# ### Curse of Dimensionality
# 
# Dimensionality in a dataset becomes a severe impediment to achieve a reasonable efficiency for most algorithms. Increasing the number of features does not always improve accuracy. When data does not have enough features, the model is likely to underfit, and when data has too many features, it is likely to overfit. Hence it is called the curse of dimensionality. The curse of dimensionality is an astonishing paradox for data scientists, based on the exploding amount of n-dimensional spaces — as the number of dimensions, n, increases.

# ![Dim](figures/curse.png)

# ### Sparseness
# 
# The sparseness of data is the property of being scanty or scattered. It lacks denseness, and its high percentage of the variable’s cells do not contain actual data. Fundamentally full of “empty” or “N/A” values.
# 
# Points in an n-dimensional space frequently become sparse as the number of dimensions grows. The distance between points will extend to grow as the number of dimensions increases.

# ![Dim](figures/Sparse_Matrix.webp)

# ### Implications of the Curse of Dimensionality
# There are few implications of the curse of dimensionality:
# 
# - Optimization problems will be infeasible as the number of features increases.
# - Due to the absolute scale of inherent points in an n-dimensional space, as n maintains to grow, the possibility of recognizing a particular point (or even a nearby point) proceeds to fall.

# ### Dimensionality Reduction
# 
# Dimensionality reduction eliminates some features of the dataset and creates a restricted set of features that contains all of the information needed to predict the target variables more efficiently and accurately.
# 
# Reducing the number of features normally also reduces the output variability and complexity of the learning process. The covariance matrix is an important step in the dimensionality reduction process. It is a critical process to check the correlation between different features.

# ### Correlation and its Measurement
# 
# There is a concept of correlation in machine learning that is called multicollinearity. Multicollinearity exists when one or more independent variables highly correlate with each other. Multicollinearity makes variables highly correlated to one another, which makes the variables’ coefficients highly unstable.
# 
# The coefficient is a significant part of regression, and if this is unstable, then there will be a poor outcome of the regression result. Multicollinearity is confirmed by using **Variance Inflation Factors** (VIF). Therefore, if multicollinearity is suspected, it can be checked using the variance inflation factor (VIF).

# $\text{VIF}_j = \frac{1}{1 - R_j^2}$
# 
# where $\text{VIF}_j$ is the VIF for the predictor variable $X_j$, and $R_j^2$ is the coefficient of determination for the regression model where $X_j$ is the response variable and all other predictor variables are used to predict $X_j$. The VIF measures the extent to which the variance of the estimated coefficient for $X_j$ is inflated due to its linear dependence on other predictor variables in the model. A high VIF indicates strong collinearity between $X_j$ and other predictors.

# ![Dim](figures/vif_plot.png)

# Rules from VIF:
# 
# - A VIF of 1 would indicate complete independence from any other variable.
# - A VIF between 5 and 10 indicates a very high level of collinearity [4].
# - The closer we get to 1, the more ideal the scenario for predictive modeling.
# - Each independent variable regresses against each independent variable, and we calculate the VIF.

# Heatmap also plays a crucial role in understanding the correlation between variables.
# 
# The type of relationship between any two quantities varies over a period of time.
# 
# Correlation varies from -1 to +1
# 
# To be precise,
# 
# - Values that are close to +1 indicate a positive correlation.
# - Values close to -1 indicate a negative correlation.
# - Values close to 0 indicate no correlation at all.

# ![Dim](figures/cov.PNG)

# A correlation from the representation of the heatmap:
# 
# - Among the first and the third features.
# - Between the first and the fourth features.
# - Between the third and the fourth features.
# 
# Independent features:
# 
# -The second feature is almost independent of the others.
# 
# Here the correlation matrix and its pictorial representation have given the idea about the potential number of features reduction. Therefore, two features can be kept, and other features can be reduced apart from those two features.
# 
# There are two ways of dimensionality reduction:
# 
# - Feature Selection
# - Feature Extraction
# 
# Dimensionality Reduction can ignore the components of lesser significance.

# ### Feature Selection
# In feature selection, usually, a subset of original features is selected.

# ![Dim](figures/feature_selection.png)

# ### Feature Extraction
# In feature extraction, a set of new features are found. That is found through some mapping from the existing features. Moreover, mapping can be either linear or non-linear.

# ![Dim](figures/ff.png)

# ### Linear Feature Extraction
# Linear feature extraction is straightforward to compute and analytically traceable.
# 
# Widespread linear feature extraction methods:
# 
# - Principal Component Analysis (PCA): It seeks a projection that preserves as much information as possible in the data.
# - Linear Discriminant Analysis (LDA):- It seeks a projection that best discriminates the data.

# ![Dim](figures/pcalda.jpeg)

# ### Principal Component Analysis (PCA)
# Principal Component Analysis (PCA) is an exploratory approach to reduce the data set's dimensionality to 2D or 3D, used in exploratory data analysis for making predictive models. Principal Component Analysis is a linear transformation of data set that defines a new coordinate rule such that:
# 
# The highest variance by any projection of the data set appears to laze on the first axis.
# The second biggest variance on the second axis, and so on.
# We can use principal component analysis (PCA) for the following purposes:
# 
# - To reduce the number of dimensions in the dataset.
# - To find patterns in the high-dimensional dataset
# - To visualize the data of high dimensionality
# - To ignore noise
# - To improve classification
# - To gets a compact description
# - To captures as much of the original variance in the data as possible
# 
# In summary, we can define principal component analysis (PCA) as the transformation of any high number of variables into a smaller number of uncorrelated variables called principal components (PCs), developed to capture as much of the data’s variance as possible.

# - 1. Math of PCA [Math of PCA](https://towardsdatascience.com/the-mathematics-behind-principal-component-analysis-fff2d7f4b643).
# - 2. More math of PCA [Math of PCA](https://medium.com/analytics-vidhya/the-math-of-principal-component-analysis-pca-bf7da48247fc).
# 
# 

# ### What is PCA
# Principal Component Analysis (PCA) is a **linear dimensionality reduction** technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.
# 
# Dimensions are nothing but features that represent the data. For example, A 28 X 28 image has 784 picture elements (pixels) that are the dimensions or features which together represent that image.
# 
# One important thing to note about PCA is that it is an **Unsupervised dimensionality reduction** technique, you can cluster the similar data points based on the feature correlation between them without any supervision (or labels)
# 
# According to Wikipedia, PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components.

# - Wiki of PCA [Wiki](https://en.wikipedia.org/wiki/Principal_component_analysis).

# 

# ### Why use PCA?
# By reducing the number of features, PCA can help:
# 
# - Reduce the risk of overfitting a model to noisy features.
# - Speed-up the training of a machine learning algorithm
# - **Make simpler data vizualisations**. When working on any data related problem, the challenge in today's world is the sheer volume of data, and the variables/features that define that data. To solve a problem where data is the key, you need extensive data exploration like finding out how the variables are correlated or understanding the distribution of a few variables. Considering that there are a large number of variables or dimensions along which the data is distributed, visualization can be a challenge and almost impossible.Hence, PCA can do that for you since it projects the data into a lower dimension, thereby allowing you to visualize the data in a 2D or 3D space with a naked eye.
# 

# For example, the Iris dataset has 4 features… hard to plot a 4D graph.

# ![Dim](figures/iris.webp)

# However, we can use PCA to reduce the number of features to 3 and plot on a 3D graph.

# ![Dim](figures/3dpca.webp)

# ![Dim](figures/2dpca.webp)

# ### How the PCA Machine Learning Algorithm Works?
# 
# PCA identifies the intrinsic dimension of a dataset.
# 
# In other words, it identifies the smallest number of features required to make an accurate prediction.
# 
# A dataset may have a lot of features, but not all features are essential to the prediction.

# ![Dim](figures/noise.webp)

# The features kept are the ones that have significant variance.
# 
# - The linear mapping of the data to a lower-dimensional space is performed in a way that maximizes the variance of the data.
# - PCA assumes that features with low variance are irrelevant and features with high variance are informative.

# ### Steps involved in PCA
# 
# - Standardize the PCA.
# - Calculate the covariance matrix.
# - Find the eigenvalues and eigenvectors for the covariance matrix.
# - Plot the vectors on the scaled data.

# ![Dim](figures/steps.jpeg)

# ### PCA for Data Visualization
# 
# For a lot of machine learning applications, it helps to visualize your data. Visualizing two- or three-dimensional data is not that challenging. However, even the Iris data set used in this part of the tutorial is four-dimensional. You can use PCA to reduce that four-dimensional data into two  or three dimensions so that you can plot, and hopefully, understand the data better.

# STEP 1: LOAD THE IRIS DATA SET
#     
# The iris data set comes with scikit-learn and doesn’t require you to download any files from some external websites. The code below will load the Iris data set.

# In[1]:


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df


# STEP 2: STANDARDIZE THE DATA
# 
# PCA is affected by scale, so you need to scale the features in your data before applying PCA. Use StandardScaler to help you standardize the data set’s features onto unit scale (mean = 0 and variance = 1), which is a requirement for the optimal performance of many machine learning algorithms. If you don’t scale your data, it can have a negative effect on your algorithm. 

# In[2]:


from sklearn.preprocessing import StandardScaler

features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target
y = df.loc[:,['target']].values
x


# In[3]:



# Standardizing the features
x = StandardScaler().fit_transform(x)
x


# STEP 3: PCA PROJECTION TO 2D
#     
# The original data has four columns (sepal length, sepal width, petal length and petal width). In this section, the code projects the original data, which is four-dimensional, into two dimensions. After dimensionality reduction, there usually isn’t a particular meaning assigned to each principal component. The new components are just the two main dimensions of variation.

# In[4]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf


# In[5]:


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf


# Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.

# STEP 4: VISUALIZE 2D PROJECTION
#     
# This section is just plotting two-dimensional data. Notice on the graph below that the classes seem well separated from each other.

# In[6]:


import matplotlib.pyplot as plt

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


# The explained variance tells you how much information (variance) can be attributed to each of the principal components. This is important because while you can convert four-dimensional space to a two-dimensional space, you lose some of the variance (information) when you do this. By using the attribute explained_variance_ratio_, you can see that the first principal component contains 72.77 percent of the variance, and the second principal component contains 23.03 percent of the variance. Together, the two components contain 95.80 percent of the information.

# In[7]:


explained_variance_ratio_


# ### PCA to Speed-Up Machine Learning Algorithms
# 
# While there are other ways to speed up machine learning algorithms, one less commonly known way is to use PCA. For this section, we aren’t using the Iris data set, as it only has 150 rows and four feature columns. The MNIST database of handwritten digits is more suitable, as it has 784 feature columns (784 dimensions), a training set of 60,000 examples and a test set of 10,000 examples.

# STEP 1: DOWNLOAD AND LOAD THE DATA
# 
# You can also add a data_home parameter to fetch_mldata to change where you download the data.

# In[4]:


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist


# The images that you downloaded are contained in mnist.data and has a shape of (70000, 784) meaning there are 70,000 images with 784 dimensions (784 features).
# 
# The labels (the integers 0–9) are contained in mnist.target. The features are 784 dimensional (28 x 28 images), and the labels are numbers from 0–9.

# STEP 2: SPLIT DATA INTO TRAINING AND TEST SETS
# 
# The code below performs a train test split which puts 6/7th of the data into a training set and 1/7 of the data into a test set.

# In[21]:


from sklearn.model_selection import train_test_split

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# STEP 3: STANDARDIZE THE DATA
# 
# The text in this paragraph is almost an exact copy of what was written earlier. PCA is affected by scale, so you need to scale the features in the data before applying PCA. You can transform the data onto unit scale (mean = 0 and variance = 1), which is a requirement for the optimal performance of many machine learning algorithms. StandardScaler helps standardize the data set’s features. You fit on the training set and transform on the training and test set.

# In[22]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# STEP 4: IMPORT AND APPLY PCA
# 
# Notice the code below has .95 for the number of components parameter. It means that scikit-learn chooses the minimum number of principal components such that 95 percent of the variance is retained.

# In[23]:


from sklearn.decomposition import PCA

# Make an instance of the Model
pca = PCA(.95)


# Fit PCA on the training set. You are only fitting PCA on the training set.

# In[24]:


pca.fit(train_img)


# You can find out how many components PCA has after fitting the model using pca.n_components_. In this case, 95 percent of the variance amounts to 330 principal components.

# STEP 5: APPLY THE MAPPING (TRANSFORM) TO THE TRAINING SET AND THE TEST SET.

# 1. Import the model you want to use.
# In sklearn, all machine learning models are implemented as Python classes.

# In[26]:


from sklearn.linear_model import LogisticRegression


# 2. Make an instance of the model.

# In[27]:


# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')


# 3. Train the model on the data, storing the information learned from the data.
# 
# The model is learning the relationship between digits and labels.

# In[28]:


logisticRegr.fit(train_img, train_lbl)


# 4. Predict the labels of new data (new images).
# 
# This part uses the information the model learned during the model training process. The code below predicts for one observation.

# In[29]:


# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))


# In[30]:


# Predict for One Observation (image)
logisticRegr.predict(test_img[0:10])


# STEP 7: MEASURING MODEL PERFORMANCE
#     
# While accuracy is not always the best metric for machine learning algorithms (precision, recall, F1 score, ROC curve, etc., would be better), it is used here for simplicity.
# 
# 

# In[31]:


logisticRegr.score(test_img, test_lbl)


# TESTING THE TIME TO FIT LOGISTIC REGRESSION AFTER PCA
# 
# The whole purpose of this section of the tutorial was to show that you can use PCA to speed up the fitting of machine learning algorithms. The table below shows how long it took to fit logistic regression on my MacBook after using PCA (retaining different amounts of variance each time).

# ![Dim](figures/6_pca-in-python.jpg)
