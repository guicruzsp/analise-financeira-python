# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:48:41 2021

In this code we will analyze stocks with Single-Factor and Multifactor models.

@author: Guilherme_C_Souza
"""

# We will use the same packages from last time and numpy
# numpy is a useful package for linear algebra
import matplotlib.pyplot as plt   # This allow us to make plots
import pandas as pd
import pandas_datareader as pd_reader
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


##### Part 1 - Single-Factor Model

# Obtain stock data and market index
stocks = ["PETR4.SA","BBAS3.SA","^BVSP"]
data   = pd_reader.get_data_yahoo(stocks, 
                       start='2000-01-01', 
                       end='2021-01-01', 
                       interval='m')['Adj Close']


data_returns = data.pct_change().dropna()

# Obtain Risk-Free rate

## BACEN dataset

# Lets define a function to import data from their database
def consulta_bc(codigo_bcb):
  url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
  df = pd.read_json(url)
  df['data'] = pd.to_datetime(df['data'], dayfirst=True)
  df.set_index('data', inplace=True)
  return df


selic = consulta_bc(1178)
selic = selic.loc['2000-01-01':'2020-12-31']

RF = selic
RF = (1+RF/100)**(1/252)
RF = RF.resample('1M').prod()
RF = round((RF-1),4)
RF.index = RF.index + pd.offsets.MonthBegin(1)

# Now we can calculate the excess returns for our stocks and our market returns
data_excess_returns = data_returns.iloc[:,0:3].sub(RF.iloc[:,0], axis = 0)
data_excess_returns = pd.DataFrame(data_excess_returns).dropna()


# Lets estimate our betas for one of our stocks (PETR4 in this case)
X = np.array(data_excess_returns.iloc[:,2]).reshape(-1, 1)
Y = np.array(data_excess_returns.iloc[:,0]).reshape(-1, 1)

model = LinearRegression()  # create object for the class
model.fit(X, Y)  # perform linear regression
Y_pred = model.predict(X)  # make predictions

plt.scatter(X, Y, label = "Realized Returns")
plt.plot(X, Y_pred, color='red', label="Expected Returns")
plt.xlabel('Market Returns')
plt.ylabel('Stock Returns')
plt.title('Expected and Realized Returns')
plt.legend(frameon= False ,loc='upper left', fontsize ='small' )
plt.show()


beta_a = model.coef_
alpha_a = model.intercept_



##### Part 2 - Multifactor Model

# Here we will add some risk factors
# In this example we will use macro factors, but we can use other variables

gdp  = consulta_bc(4380)

RF_change = RF.pct_change().dropna()
gdp_change = gdp.pct_change().dropna()

risk_factors = pd.concat([data_excess_returns.iloc[:,2],gdp_change,RF_change], axis=1).dropna()



# Multifactor model for BBAS3
X = np.array(risk_factors).reshape(-1, 3)
Y = np.array(data_excess_returns.iloc[:,1]).reshape(-1, 1)

model = LinearRegression()  # create object for the class
model.fit(X, Y)  # perform linear regression
Y_pred = model.predict(X)  # make predictions

plt.scatter(X[:,0], Y, label = "Realized Returns")
plt.plot(X[:,0], Y_pred, color='red', label="Expected Returns")
plt.xlabel('Market Returns')
plt.ylabel('Stock Returns')
plt.title('Expected and Realized Returns')
plt.legend(frameon= False ,loc='upper left', fontsize ='small' )
plt.show()


beta_a = model.coef_
alpha_a = model.intercept_

model.score(X, Y)


# Using statsmodels to get the standard errors

X_with_intercept = np.empty(shape=(len(X), 4), dtype=np.float)
X_with_intercept[:, 0] = 1
X_with_intercept[:, 1:4] = X

ols = sm.OLS(Y, X_with_intercept)
ols_result = ols.fit()
ols_result.summary()








