# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 19:33:54 2021

In this code we will build a portfolio using 2 risky assets and n risky assets.

@author: Guilherme_C_Souza
"""
# We will use the same packages from last time and numpy
# numpy is a useful package for linear algebra
import matplotlib.pyplot as plt   # This allow us to make plots
import pandas as pd
import pandas_datareader as pd_reader
import numpy as np


##### Part 1: 2 risky assets

# Lets obtain stock data for two assets
stocks = ["PETR4.SA","BBAS3.SA"]
data   = pd_reader.get_data_yahoo(stocks, 
                       start='2000-01-01', 
                       end='2021-01-01', 
                       interval='m')['Adj Close']

# Lets obtain the stock returns
data_returns = data.pct_change().dropna()

# Returns for asset 1
R_1 = data_returns.iloc[:,0]
R_1_bar = R_1.mean()
# Returns for asset 2
R_2 = data_returns.iloc[:,1]
R_2_bar = R_2.mean()

# Now we obtain the statistics of our assets

# standard deviation of asset 1
sigma_1 = R_1.std()
# standard deviation of asset 2
sigma_2 = R_2.std()
# correlation between assets
rho_1_2 = data_returns.corr().iloc[0,1]

# By using the formula seem in class we can
# obtain the global minimum-variance (GMV) portfolio

# Share of asset 1
x_1 = (sigma_2**2 - sigma_1*sigma_2*rho_1_2) / (sigma_1**2 + sigma_2**2 - 2*sigma_1*sigma_2*rho_1_2)
# Share of asset 2
x_2 = (sigma_1**2 - sigma_1*sigma_2*rho_1_2) / (sigma_1**2 + sigma_2**2 - 2*sigma_1*sigma_2*rho_1_2)


# Now we can calculate our portfolio return
R_portfolio = x_1*R_1 + x_2*R_2

R_portfolio_bar = x_1*R_1_bar + x_2*R_2_bar # alternatively: (R_portfolio_bar = R_portfolio.mean())

sigma_p = (x_1**2 * sigma_1**2 + x_2**2 * sigma_2**2 + 2*x_1*x_2*sigma_1*sigma_2*rho_1_2)**(1/2) # alternatively: (sigma_p = R_portfolio.std())


# Lets compare the individual stocks and the portfolio
column_names = ['Asset 1', 'Asset 2', 'Portfolio']
row_names    = ['Stdev', 'Returns']
comparing_table = pd.DataFrame([[sigma_1,sigma_2,sigma_p],[R_1_bar,R_2_bar,R_portfolio_bar]],
                               columns = ['Asset 1', 'Asset 2', 'Portfolio'],
                               index = ['Stdev', 'Returns'])


fig = plt.figure()
#plt.plot(stds, means, 'o', markersize=1)
p1 = plt.plot(R_1, 'o', markersize=1, label = "PETR")
p2 = plt.plot(R_2, 'o', markersize=1, color = "red", label = "BBAS")
p3 = plt.plot(R_portfolio, 'o', markersize=1, color = "green", label = "Portfolio")
plt.legend(frameon= False ,loc='upper right', fontsize ='small' )
plt.xlabel('std')
plt.ylabel('Return')
plt.title('Efficient Frontier and Individual Assets')
plt.show()


##### Part 2: N risky assets

# Lets obtain stock data for N assets (were we use 15 without loss of generality)
stocks = ["PETR4.SA","BBAS3.SA","ABEV3.SA","BBSE3.SA","BBDC3.SA",
          "GOLL4.SA","ELET3.SA","LREN3.SA","MGLU3.SA","ITUB4.SA",
          "SBSP3.SA","USIM5.SA","VALE3.SA","WEGE3.SA","CCRO3.SA"]
data   = pd_reader.get_data_yahoo(stocks, 
                       start='2015-01-01', 
                       end='2021-01-01', 
                       interval='m')['Adj Close']

# Lets obtain the stock returns
data_returns = data.pct_change().dropna()

# We will use the formula from class
number_of_assets = data_returns.shape[1]

# Create a covariance matrix
cov_matrix = data_returns.cov()

# Mean return of each asset
R_bar = data_returns.mean()


# Apply the formula to find the weights for a portfolio with monthly expected return of 0.03
mu_1 = np.matrix([R_bar,np.ones(number_of_assets)])
mu_1_t = mu_1.transpose()
inverse_V = np.linalg.inv(cov_matrix)
R_P_1 = np.matrix([[0.03], [1]])

# Applying the formula
weights =  inverse_V @ mu_1_t @ np.linalg.inv(mu_1 @ inverse_V @ mu_1_t) @ R_P_1
sigma_p = ((weights.transpose() @ np.matrix(cov_matrix) @ weights)[0,0])**(1/2)


## We may also solve for the Global Minimum Variance Portfolio
vector_1 = np.ones(number_of_assets)

weights_star = (inverse_V @ vector_1)/(vector_1.transpose() @ inverse_V @ vector_1)
mu_star      = (vector_1.transpose() @ inverse_V @ R_bar) / (vector_1.transpose() @ inverse_V @ vector_1)
sigma_star   = np.sqrt((vector_1.transpose() @ inverse_V @ vector_1)**(-1))




# Plot Efficient Frontier 
efficient_frontier = np.empty([11,2])
count = 0
for i in np.linspace(0,0.1,11):
    R_P_1 = np.matrix([[i], [1]])
    weights =  inverse_V @ mu_1_t @ np.linalg.inv(mu_1 @ inverse_V @ mu_1_t) @ R_P_1
    sigma_p = ((weights.transpose() @ np.matrix(cov_matrix) @ weights)[0,0])**(1/2)
    ret = np.matrix(R_bar).dot(weights)
    efficient_frontier[count,0] = ret
    efficient_frontier[count,1] = sigma_p
    
    count = count + 1


assets_table = np.matrix([R_bar,data_returns.std()]).T

fig = plt.figure()
#plt.plot(stds, means, 'o', markersize=1)
p1 = plt.plot(assets_table[:,1], assets_table[:,0], 'o', markersize=1, label = "Stocks")
p2 = plt.plot(efficient_frontier[:,1], efficient_frontier[:,0], 'o', markersize=3, color = "red", label = "Efficient frontier")
p3 = plt.plot(sigma_star, mu_star, 'o', markersize=4, color = "green", label = "Min Var Portfolio")
plt.legend(frameon= False ,loc='upper right', fontsize ='small' )
plt.xlabel('std')
plt.ylabel('Return')
plt.title('Efficient Frontier and Individual Assets')
plt.xlim([0, 0.25])
plt.ylim([-0.05, 0.15])
plt.show()









