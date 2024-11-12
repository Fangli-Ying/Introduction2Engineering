#!/usr/bin/env python
# coding: utf-8

# # What is Machine Learning, and how does it work? ([video #1](https://www.youtube.com/watch?v=HcqpanDadyQ))

# ## Last week class
# 
# - What is Machine Learning?
# - What are the main categories of Machine Learning?
# 
# ## Agenda
# - What are some examples of Machine Learning?
# - How does Machine Learning "work"?
# 
# - What are the benefits and drawbacks of scikit-learn?
# - How do I install scikit-learn?
# - How do I use the Jupyter Notebook?
# - What are some good resources for learning Python?

# ## Review of ML to the data
# - **Knowledge from data**: Starts with a question that might be answerable using data
# - **Automated extraction**: A computer provides the insight
# - **Semi-automated**: Requires many smart decisions by a human

# ## What are the main categories of Machine Learning?
# 
# **Supervised learning**: Making predictions using data(Predictive model)
#     
# - Example: Is a given email "spam" or "non spam"?
# - There is a specific outcome we are trying to predict

# ![Spam filter](img/01_spam_filter.png)

# **Unsupervised learning**: Extracting structure from data or learning the representation
# 
# - Example: Segment grocery store shoppers into clusters that exhibit similar behaviors
# - There is no "right answer"
# - e.g. how many clusters are there, which clutser the data belong with or how to describe the cluster

# ([Example #1](https://www.kaggle.com/c/titanic))
# - The goal is to predict if a passenger survived the sinking of the Titanic or not.
# - For each in the test set, you must predict a 0 or 1 value for the variable.
# - Is this a supervised learning or unsupervised learning?

# ## How does Machine Learning "work"?
# 
# High-level steps of supervised learning:
# 
# 1. First, train a **Machine Learning model** using **labeled data**
# 
#     -"Labeled data" has been labeled with the outcome (Can you give an example of labeling spam email?)
#     -"Machine Learning model" learns the relationship between the attributes of the data and its outcome
# 
# 2. Then, make **predictions** on **new data** for which the label is unknown

# ## The workflow of ML
# ![Supervised learning](img/01_supervised_learning.png)

# The primary goal of supervised learning is to build a model that "generalizes": It accurately predicts the **future** rather than the **past**!

# ## Questions about Machine Learning
# 
# - How do I choose **which attributes** of my data to include in the model?
# - How do I choose **which model** to use?
# - How do I **optimize** this model for best performance?
# - How do I ensure that I'm building a model that will **generalize** to unseen data?
# - Can I **estimate** how well my model is likely to perform on unseen data?

# ## Resources
# 
# - Book: [An Introduction to Statistical Learning](https://www.statlearning.com/) (section 2.1, 14 pages)
# - Video: [Learning Paradigms](https://www.youtube.com/watch?v=mbyG85GZ0PI&t=2162s) (13 minutes, starting at 36:02)

# ## Comments or Questions?
# 
# 
# - Website and Email: https://fangli-ying.github.io
