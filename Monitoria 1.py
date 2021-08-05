# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 21:15:14 2021

In this code we will import stock data from yahoo finance using pandas_datareader and 
also to import data from BACEN ( https://www3.bcb.gov.br/sgspub )


@author: Guilherme_C_Souza
"""

# If this is your first time using Python, you will need to install the pandas_datareader package
# The package allows us to import stock data from yahoo finance
# pip install pandas_datareader

# Import pandas_datareader and matplotlib
import matplotlib.pyplot as plt   # This allow us to make plots
import pandas as pd
import pandas_datareader as pd_reader


##### Part 1 - Stock data

# Get the data for the stocks by specifying the stock ticker, start date, and end date 
stocks = ["PETR4.SA","VALE3.SA","ITUB3.SA"]
data   = pd_reader.get_data_yahoo(stocks, 
                       start='2000-01-01', 
                       end='2021-01-01', 
                       interval='m')['Adj Close']



# Lets look at what we got
data.head()
data.columns

# Plot the adjusted prices 
data.iloc[:,0:3].plot() 
plt.title("Stock Prices")
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()

# Lets calculate stock returns
data = data.iloc[:,0:3].dropna()
data_returns = data.pct_change().dropna()

# Plot the returns 
data_returns.plot() 
plt.title("Returns")
plt.xlabel('Data')
plt.ylabel('Returns')
plt.show()

# Plot the histogram of returns
data_returns.hist(bins = 20)
plt.title("Histogram of returns")
plt.show()


##### Part 2 - BACEN data

# Here we will import IPCA and SELIC data

# Lets define a function to import data from their database (This function was obtained from https://colab.research.google.com/drive/1_t6-vO_Mv1Iv_4Wykexeb81FJkOPY-CI)
def consulta_bc(codigo_bcb):
  url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json'.format(codigo_bcb)
  df = pd.read_json(url)
  df['data'] = pd.to_datetime(df['data'], dayfirst=True)
  df.set_index('data', inplace=True)
  return df

# We can retrieve the code for each series by looking at the website ( https://www3.bcb.gov.br/sgspub )
ipca  = consulta_bc(433)
selic = consulta_bc(1178)

# Lets restrict our analysis to the period after 2000
ipca  = ipca.loc['2000-01-01':'2020-12-31']
selic = selic.loc['2000-01-01':'2020-12-31']

ipca.plot()
plt.title("IPCA monthly")
plt.xlabel('Data')
plt.ylabel('Taxa')
selic.plot()
plt.title("Selic")
plt.xlabel('Data')
plt.ylabel('Meta')


##### Part 3 - Data analysis

# Lets first obtain the arithmetic and geometric average of returns from PETR4
# Arithmetic average of returns
AAR = data_returns.iloc[:,0].sum()/len(data_returns)
# Geometric average of returns
GAR = (data_returns.iloc[:,0]+1).prod() ** (1/len(data_returns)) - 1

# Lets compare our results
print("Arithmetic average of returns: ",round(AAR*100,2),"%")
print("Geometric average of returns: ",round(GAR*100,2),"%")

# Now, lets obtain the excess returns for our assets 
# Using the risk-free rate we just obtained

# Since our data was downloaded at a daily frequency and
# An annualized rate
# Lets convert it to monthly
RF = selic
RF = (1+RF/100)**(1/252)
RF = RF.resample('1M').prod()
RF = round((RF-1),4)
RF.index = RF.index + pd.offsets.MonthBegin(1)

# Now we can calculate the excess returns for PETR4
stock_excess_returns = data_returns.iloc[:,0] - RF.iloc[:,0]
stock_excess_returns = pd.DataFrame(stock_excess_returns)
stock_excess_returns.plot()
plt.title("Exces Returns")
plt.xlabel('Data')
plt.ylabel('Taxa')
stock_excess_returns.hist(bins = 20)

# Next we will look at the sharpe ratio for our asset
# on the analyzed period
Sharpe_ratio = (stock_excess_returns.mean())/(stock_excess_returns.std())

# Lets plot the excess returns and the RF rate for each month
data_ER_RF = pd.concat([stock_excess_returns,RF], axis=1)
data_ER_RF.plot()
plt.title("Excess Returns and Risk-Free rate")
plt.xlabel('Data')
plt.ylabel('Taxa')


# At last, we will look at the real interest rates in Brazil for 
# The period we are analyzing
inflation_rate = ipca/100
nominal_interest_rate = RF

real_interest_rate = (1+nominal_interest_rate)/(1+inflation_rate) - 1
(real_interest_rate*100).plot()
plt.title("Monthly real interest rate")
plt.xlabel('Data')
plt.ylabel('Taxa')

annualized_real_interest_rate = (1+real_interest_rate)**12-1
(annualized_real_interest_rate*100).plot()
plt.title("Annualized real interest rate")
plt.xlabel('Data')
plt.ylabel('Taxa')

