### Boston Housing Dataset Report

## Project Description

You want to be the best real estate agent out there. In order to compete with other agents in your area, you decide to use machine learning. You are going to use various statistical analysis tools to build the best model to predict the value of a given house. Your task is to find the best price your client can sell their house at. The best guess from a model is one that best generalizes the data.

For this assignment your client has a house with the following feature set: [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]. To get started, use the example scikit implementation. You will have to modify the code slightly to get the file up and running.

## Statistical Analysis and Data Exploration

Loading the dataset:

```python
from sklearn import datasets

city_data = datasets.load_boston()

# Get the labels and features from the housing data
housing_prices = city_data.target
housing_features = city_data.data

```

Let us know begin with the exploration of data using numpy

- Number of data points (houses)?

```python
number_of_houses = housing_features.shape[0]
print "number of houses:",number_of_houses
```
number of houses: 506

---

- Number of features?
```python
number_of_features = housing_features.shape[1]
print "number of features:",number_of_features
```
---

number of features: 13

- Minimum and maximum housing prices?
```python
max_price = np.max(housing_prices)
min_price = np.min(housing_prices)

print "max price of house:",max_price
print "min price of house:",min_price
```
max price of house: 50.0
min price of house: 5.0

---

- Mean and median Boston housing prices?
```python
mean_price = np.mean(housing_prices)
median_price = np.median(housing_prices)

print "mean price of house:",mean_price
print "median price of house:",median_price
```
mean price of house: 22.5328063241
median price of house: 21.2

---

- Standard deviation?
```python
standard_deviation = np.std(housing_prices)
print "standard deviation for prices of house:",standard_deviation
```
standard deviation for prices of house: 9.18801154528

## Evaluating Model Performance

- Which measure of model performance is best to use for predicting Boston housing data and analyzing the errors? Why do you think this measurement most appropriate? Why might the other measurements not be appropriate here?

I believe the best measure of performance would be the R-squared measure. I choose this over mean squarred error beacause I belive in the housing industry, prices seem to flucutuate erratically. Thus, rather than looking for a tight fit, it would be intuitive to look for a more relative fit. Other measurement which include median deviation and absolute error would not emphasize large errors like in the case of MSE and R-squared measure.

---


- Why is it important to split the Boston housing data into training and testing data? What happens if you do not do this?

One of the biggest problems that could occur is overfitting. Hence, we should always split the data into training and testing. This would ensure that we don't cause a tight fit which could ultimately lead to a dismayal performance on the testing dataset.

---

- What does grid search do and why might you want to use it?

Grid search helps in parameter tuning and the selecetion of appropriate model based on the parameters we wish to tune. While computing this, it also implemets cross validation folds, thus eliminating the risk of overfitting. It also has the flexibility of making a customised scorer function.

---

- Why is cross validation useful and why might we use it with grid search?

The goal of cross validation is to define a dataset to "test" the model in the training phase (i.e., the validation dataset), in order to limit problems like overfitting, give an insight on how the model will generalize to an independent dataset (i.e., an unknown dataset, for instance from a real problem), etc.

Gird search performs 3-fold cross validation, hence we use it.

##  Analyzing Model Performance

- Look at all learning curve graphs provided. What is the general trend of training and testing error as training size increases?

As the training size increases, the error reduces and the model begins to predict better. This can be seen form the learning curves. With the increase in training size, the trainign error is almost 0 whereas the testing error continues to reduce. Thus, there is improvement in the regression model as training size increases.

--- 

- Look at the learning curves for the decision tree regressor with max depth 1 and 10 (first and last learning curve graphs). When the model is fully trained does it suffer from either high bias/underfitting or high variance/overfitting?

I think we are underfitting. When we observe the mean squarred error, it fluctuates almost as if it looks like a cycle. This is a clear indication that a linear function is not sufficient for explaining the data.

---

- Look at the model complexity graph. How do the training and test error relate to increasing model complexity? Based on this relationship, which model (max depth) best generalizes the dataset and why?

I think a max_depth between 5-10 generalises the dataset the best. It doesn't overfit as in the case of higher depths, we simply get a training error of 0. This means we are completly fitting the data and there is ahigh chance of overfitting.


## Model Prediction

- Model makes predicted housing price with detailed model parameters (max depth) reported using grid search. Note due to the small randomization of the code it is recommended to run the program several times to identify the most common/reasonable price/model complexity.

```python

Final Model:

GridSearchCV(cv=None, error_score='raise',
       estimator=DecisionTreeRegressor(criterion='mse',
       	 max_depth=None, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1,
            min_samples_split=2,
           min_weight_fraction_leaf=0.0, random_state=None,
           splitter='best'),
       fit_params={}, iid=True, loss_func=None, n_jobs=1,
       param_grid={'min_samples_split': (1, 2, 3),
        'max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
         'min_samples_leaf': (1, 2, 3)},
       pre_dispatch='2*n_jobs', refit=True, score_func=None, scoring='r2',
       verbose=0)

House: [11.95, 0.0, 18.1, 0, 0.659, 5.609, 90.0,
 1.385, 24, 680.0, 20.2, 332.09, 12.13]

Prediction: [ 20.96776316]

```

---

- Compare prediction to earlier statistics and make a case if you think it is a valid model.

The central tendency for the given dataset with respect to the mean and the median are as follows:
mean price of house: 22.5328063241
median price of house: 21.2

This means that our current prediction for the given house is credible. Therefore we can say that we have a model that fits the data to an extent.

However, I would say that the model still doesn't completely describe the variance in the dataset. If we plot the graph of residuals, we get a cycle like graph. This means that a linear mode won't be able to generalise the data. We are underfitting the dataset.

![Image of Residual Plot]
(residual_plot.png)