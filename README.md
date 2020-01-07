# Linear-Regression-Model
Quick Start Guide to using Linear Regression Models in Python  
## Introduction  
A linear regression model might be the simplest of all models to understand. It tries to explain a relationship between  
independant variables and a dependent variable. We will do this by using the least squared method. This is a statistical   procedure to find the best fit for a set of data points by minimizing the sum of the offsets or residuals of points from the   plotted curve. Least squares regression is used to predict the behavior of dependent variables.  
 
## Prerequisites
1. Python 3.7 or newer  
2. Scikit-Learn module for Python  
3. Pandas module for Python  
4. Numpy module for Python  
5. Matplotlib module for Python  
6. Seaborn module for Python
  
## Walkthough  
Start by importing all modules you will need at the beginning of your script. These include: Pandas, Scikit-Learn,  
Matplotlib, and seaborn.  
```
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```  
Next import the csv file and take a closer look. Below we are importing the csv file as an object "df".  
Once again we print the head of the data along with the info to get an idea of what the data looks like.  

```
df = pd.read_csv(r"D:\Datasets\USA_Housing.csv")
print(df.head())
print(df.info())
```  
Make a pairplot using seaborn to compare each variable to one other. this will show us if there is any correlation between  
features. To see if the data is evenly distributed, we can make a distribution plot of the price. This data appars to be evenly  
distributed.
```
sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])
plt.show()
```  


The x and y variables can now be established. The X variables are going to be the features you would like to use to   
predict y. Y is going to be the dependant variable. In this case, the y variable is "Price".  
```
X= df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y=df['Price']
```  
Now that the variables have been established, it is time to split the data into two parts. The training set  
is what our model is going to look at and learn from. The model will then try to apply what it has learned to the test data.  
The "test_size" argument, is set to 0.4 in this example. That means that 60% of the data will be used to train the model  
and the remaining 40% will be used to test how accurate the model is. The "random_state" parameter sets the random  
number generator. This makes it so we are able to replicate results.  
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) 
```
The model can now be instantiated. Call the linear regression model and assign it to lm. Next the model needs to be fitted  
or trained as shown below. We can also check the coefficient of each feature below to see which features play the biggest  
role in price change of houses. This data is synthetically generated so it might not look accurate, but it is a worth while  
exercise. Next the perdictions need to be generated using the .predict() method.  
```
m = LinearRegression() #creating an instanciation of the linear regression model
lm.fit(X_train,y_train) #this is attempting to teach the algo based on the training data set that we pulled

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff']) # as follows -- the data we want to show (the coef), for what?(each of the categories in the independant variable),name the column ('Coeff')
print(cdf)

predictions = lm.predict(X_test) #setting the prediction variable equal to the x test

```
For a visual of the correlation between the features and result, a scatter plat can be made. The linear pattern
of the data points is a good sign and means that the features do have correlation to the result.  

Lastly, it is time to analyize the accuracy of the model. This will be done using Mean Absolute Error, Mean Squared Error,  
and Root Mean Squared.  
```
print("Mean Absolute Error")
print(metrics.mean_absolute_error(y_test,predictions))
print("Mean Squared Error")
print(metrics.mean_squared_error(y_test,predictions))
print("Root Mean Squared Error")
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))

```




