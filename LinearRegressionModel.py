import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv(r"D:\Datasets\USA_Housing.csv")
print(df.head())
print(df.info())

sns.pairplot(df)
plt.show()

sns.distplot(df['Price'])
plt.show()


X= df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']] # the data that is being fed into the algo so it can guess the up coming prices

y=df['Price'] #what we would like to compare to the rest of the data, the dependant variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101) #setting each variable and then termining what percentage of the data will go as test, random state determines which numbers are being split randomingly


lm = LinearRegression() #creating an instanciation of the linear regression model
lm.fit(X_train,y_train) #this is attempting to teach the algo based on the training data set that we pulled

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff']) # as follows -- the data we want to show (the coef), for what?(each of the categories in the independant variable),name the column ('Coeff')
print(cdf)

predictions = lm.predict(X_test) #setting the prediction variable equal to the x test

plt.scatter(y_test,predictions) #comparing the y_test to the x_test to see if there is a correlation between the variables put in and the outcome
plt.title("Comparing y-test to predictions")
plt.show()



print("Mean Absolute Error")
print(metrics.mean_absolute_error(y_test,predictions))
print("Mean Squared Error")
print(metrics.mean_squared_error(y_test,predictions))
print("Root Mean Squared Error")
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
