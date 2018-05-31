# Python机器学习算法速查



## 常见的机器学习算法

以下是最常用的机器学习算法，大部分数据问题都可以通过它们解决：

1. 线性回归 (Linear Regression)
2. 逻辑回归 (Logistic Regression)
3. 决策树 (Decision Tree)
4. 支持向量机（SVM）
5. 朴素贝叶斯 (Naive Bayes)
6. K邻近算法（KNN）
7. K-均值算法（K-means）
8. 随机森林 (Random Forest)
9. 降低维度算法（Dimensionality Reduction Algorithms）
10. Gradient Boost和Adaboost算法

![这里写图片描述](https://img-blog.csdn.net/20170515234546090?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

图1：主要是对sklearn中的主要方法进行分类

![这里写图片描述](https://img-blog.csdn.net/20170315222114487?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
图2：分别对降维和参数查找的方法进行列举 
![这里写图片描述](https://img-blog.csdn.net/20170315223408740?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

图3：常用数据预处理方法 
![这里写图片描述](https://img-blog.csdn.net/20170324233958599?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## 1.线性回归 (Linear Regression)

```python
#Import Library
#Import other necessary libraries like pandas, numpy...
from sklearn import linear_model
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be 
numeric and numpy arrays

x_train=input_variables_values_training_datasets
y_train=target_variables_values_training_datasets
x_test=input_variables_values_test_datasets

# Create linear regression object
linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
linear.fit(x_train, y_train)
linear.score(x_train, y_train)

#Equation coefficient and Intercept
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)

#Predict Output
predicted= linear.predict(x_test)123456789101112131415161718192021222324
```

## 2.逻辑回归 (Logistic Regression)

```python
#Import Library
from sklearn.linear_model import LogisticRegression
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create logistic regression object

model = LogisticRegression()

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Equation coefficient and Intercept
print('Coefficient: \n', model.coef_)
print('Intercept: \n', model.intercept_)

#Predict Output
predicted= model.predict(x_test)123456789101112131415161718
```

## 3.决策树 (Decision Tree)

```python
#Import Library
#Import other necessary libraries like pandas, numpy...

from sklearn import tree
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create tree object 
model = tree.DecisionTreeClassifier(criterion='gini') # for classification, here you can change the algorithm as gini or entropy (information gain) by default it is gini  

# model = tree.DecisionTreeRegressor() for regression

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(x_test)1234567891011121314151617
```

## 4.支持向量机（SVM）

```python
#Import Library
from sklearn import svm
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object 

model = svm.SVC() # there is various option associated with it, this is simple for classification. You can refer link, for mo# re detail.

# Train the model using the training sets and check score
model.fit(X, y)
model.score(X, y)

#Predict Output
predicted= model.predict(x_test)12345678910111213
```

## 5.朴素贝叶斯 (Naive Bayes)

```
#Import Library
from sklearn.naive_bayes import GaussianNB
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset

# Create SVM classification object 
model = GaussianNB() # there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link

# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(x_test)123456789101112
```

## 6.K邻近算法（KNN）

```
#Import Library
from sklearn.neighbors import KNeighborsClassifier

#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create KNeighbors classifier object 
model = KNeighborsClassifier(n_neighbors=6) # default value for n_neighbors is 5

# Train the model using the training sets and check score
model.fit(X, y)

#Predict Output
predicted= model.predict(x_test)123456789101112
```

## 7.K-均值算法（K-means )

```
#Import Library
from sklearn.cluster import KMeans

#Assumed you have, X (attributes) for training data set and x_test(attributes) of test_dataset
# Create KNeighbors classifier object model 
model = KMeans(n_clusters=3, random_state=0)

# Train the model using the training sets and check score
model.fit(X)

#Predict Output
predicted= model.predict(x_test)123456789101112
```

## 8.随机森林 (Random Forest)

```
#random forest
#import library
from sklearn.ensemble import  RandomForestClassifier
#assumed you have x(predictor)and y(target) for training data set and x_test(predictor)of test_dataset
#create random forest object
model=RandomForestClassifier()
#train the model using the training sets and chek score
model.fit(x,y)
#predict output
predict=model.presort(x_test)12345678910
```

## 9.降低维度算法（Dimensionality Reduction Algorithms）

```
#Import Library
from sklearn import decomposition
#Assumed you have training and test data set as train and test
# Create PCA obeject 
pca= decomposition.PCA(n_components=k) #default value of k =min(n_sample, n_features)
# For Factor analysis
#fa= decomposition.FactorAnalysis()
# Reduced the dimension of training dataset using PCA

train_reduced = pca.fit_transform(train)

#Reduced the dimension of test dataset
test_reduced = pca.transform(test)12345678910111213
```

## 10.Gradient Boost和Adaboost算法

```
#Import Library
from sklearn.ensemble import GradientBoostingClassifier
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create Gradient Boosting Classifier object
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

# Train the model using the training sets and check score
model.fit(X, y)
#Predict Output
predicted= model.predict(x_test)12345678910
```

**以下实例中predict数据时为了验证其拟合度，采用的是训练集数据作为参数，实际中应该采用的是测试集，不要被误导了！！！**

![这里写图片描述](https://img-blog.csdn.net/20170220231837529?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220231925164?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220231945587?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232013509?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232032104?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232047108?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232104636?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232126324?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232140371?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
![这里写图片描述](https://img-blog.csdn.net/20170220232214218?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232306498?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](https://img-blog.csdn.net/20170220232333188?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](https://img-blog.csdn.net/20170220232349860?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232620455?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](https://img-blog.csdn.net/20170220232647580?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

![这里写图片描述](https://img-blog.csdn.net/20170220232724206?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast) 
![这里写图片描述](https://img-blog.csdn.net/20170220232738190?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQveGlhb2Rvbmd4aWV4aWU=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)