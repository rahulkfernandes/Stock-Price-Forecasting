import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from yahoofinancials import YahooFinancials

START = '2019-01-01'
END = '2022-11-30'
TICKER = 'AAPL'

def get_data(ticker, start_date, end_date):
    yahoo_financials = YahooFinancials(ticker)
    price_data = yahoo_financials.get_historical_price_data(start_date, end_date, 'daily')
    df = pd.DataFrame(price_data[ticker]['prices'])
    df.drop(['date', 'adjclose', 'formatted_date', 'high', 'low', 'open'], axis=1, inplace=True)
    return df

def plot_ma(ticker, train_df):
    ma100 = train_df['close'].rolling(100).mean()
    ma200 = train_df['close'].rolling(200).mean()
    plt.figure(figsize = (15,5))
    plt.plot(train_df['close'], label='Prices')
    plt.plot(ma100, 'r', label='100 Day Moving Average')
    plt.plot(ma200, 'g', label='200 Day Moving Average')
    plt.title(f'{ticker} Closing Price Chart')
    plt.ylabel('CLosing Price')
    plt.xlabel('Days')
    plt.legend(loc='upper left')
    plt.show()

def plot_train_test(ticker, df):
    plt.figure(figsize = (15,5))
    plt.plot(df['close'], label='All Closing Prices')
    plt.plot(df['close'][:-future_days], label='Training Data')
    plt.title(f"{ticker} Train-Test Split")
    plt.xlabel("Days")
    plt.ylabel("Closing Price USD")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    prices = get_data(TICKER, START, END)
    #plot_ma(TICKER, prices)

    future_days = 15
    prices['prediction'] = prices[['close']].shift(-future_days)
    X_train = np.array(prices.drop(['prediction'], axis=1)[:-future_days*2])
    y_train = np.array(prices['prediction'][:-future_days*2])
    #plot_train_test(TICKER, prices)

    tree = DecisionTreeRegressor().fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    knn = KNeighborsRegressor().fit(X_train, y_train)

    x_future = prices.drop(['prediction'], axis=1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    y_future = np.array(prices['close'])[-future_days:]

    tree_pred = tree.predict(x_future)
    lr_pred = lr.predict(x_future)
    knn_pred = knn.predict(x_future)
    
    tree_mse = mean_squared_error(y_future, tree_pred)
    tree_score = tree.score(x_future, y_future)
    lr_mse = mean_squared_error(y_future, lr_pred)
    lr_score = lr.score(x_future, y_future)
    knn_mse = mean_squared_error(y_future, knn_pred)
    knn_score = knn.score(x_future, y_future)
    scores = pd.DataFrame({'Models':['Decision Tree Regressor', 'Linear Regression', 'KNN Regressor'], 
        'Score':[tree_score, lr_score, knn_score], 'RMSE':[math.sqrt(tree_mse), math.sqrt(lr_mse), math.sqrt(knn_mse)]})
    print(scores)

    validation = prices[['close']][-future_days:]
    validation['lr_pred'] = lr_pred
    validation['knn_pred'] = knn_pred
    validation['tree_pred'] = tree_pred
    
    plt.style.use("bmh")
    plt.figure(figsize=(15,5))
    plt.title(f'{TICKER} Stock Price Forecasting')
    plt.xlabel("Days")
    plt.ylabel("Closing Price USD")
    #plt.plot(prices['close'], label='Closing Price')
    plt.plot(validation['close'], 'g', label='Original Price')
    plt.plot(validation['tree_pred'], 'r', label='Decision Tree Regressor')
    plt.plot(validation['knn_pred'], label='KNN Regressor')
    plt.plot(validation['lr_pred'], 'y', label='Linear Regression')
    plt.legend()
    plt.show()