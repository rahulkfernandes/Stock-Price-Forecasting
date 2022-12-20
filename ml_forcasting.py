import numpy as np
import pandas as pd
import math
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 
from yahoofinancials import YahooFinancials
import time

START = '2020-12-01'
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
    
    x_future = prices.drop(['prediction'], axis=1)[:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)
    y_future = np.array(prices['close'])[-future_days:]

    start_time = time.time()
    tree = DecisionTreeRegressor().fit(X_train, y_train)
    tree_pred = tree.predict(x_future)
    tree_timer = time.time() - start_time
    tree_rmse = math.sqrt(mean_squared_error(y_future, tree_pred))
    tree_score = tree.score(x_future, y_future)

    start_time = time.time()
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(x_future)
    lr_timer = time.time() - start_time
    lr_rmse = math.sqrt(mean_squared_error(y_future, lr_pred))
    lr_score = lr.score(x_future, y_future)

    start_time = time.time()
    knn = KNeighborsRegressor().fit(X_train, y_train)
    knn_pred = knn.predict(x_future)
    knn_timer = time.time() - start_time
    knn_rmse = math.sqrt(mean_squared_error(y_future, knn_pred))
    knn_score = knn.score(x_future, y_future)

    start_time = time.time()
    el = ElasticNet().fit(X_train, y_train)
    el_pred = el.predict(x_future)
    el_timer = time.time() - start_time
    el_rmse = math.sqrt(mean_squared_error(y_future, el_pred))
    el_score = el.score(x_future, y_future)

    start_time = time.time()
    ada = AdaBoostRegressor().fit(X_train, y_train)
    ada_pred = ada.predict(x_future)
    ada_timer = time.time() - start_time
    ada_rmse = math.sqrt(mean_squared_error(y_future, ada_pred))
    ada_score = ada.score(x_future, y_future)

    start_time = time.time()
    bay = BayesianRidge().fit(X_train, y_train)
    bay_pred = bay.predict(x_future)
    bay_timer = time.time() - start_time
    bay_rmse = math.sqrt(mean_squared_error(y_future, bay_pred))
    bay_score = ada.score(x_future, y_future)

    start_time = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(128,2), activation='identity').fit(X_train, y_train)
    mlp_pred = mlp.predict(x_future)
    mlp_timer = time.time() - start_time
    mlp_rmse = math.sqrt(mean_squared_error(y_future, mlp_pred))
    mlp_score = mlp.score(x_future, y_future)
    
    scores = pd.DataFrame({
        'Models':['DecisionTree', 'Linear', 'KNN', 'ElasticNet', 'AdaBoost', 'BaysianRidge', 'MLP'], 
        'Score':[tree_score, lr_score, knn_score, el_score, ada_score, bay_score, mlp_score], 
        'RMSE':[tree_rmse, lr_rmse, knn_rmse, el_rmse, ada_rmse, bay_rmse, mlp_rmse],
        'TimeTaken':[tree_timer, lr_timer, knn_timer, el_timer, ada_timer, bay_timer, mlp_timer]})
    print(scores)

    validation = prices[['close']][-future_days:]
    validation['lr_pred'] = lr_pred
    validation['knn_pred'] = knn_pred
    validation['tree_pred'] = tree_pred
    validation['el_pred'] = el_pred
    validation['ada_pred'] = ada_pred
    validation['bay_pred'] = bay_pred
    validation['mlp_pred'] = mlp_pred
    
    plt.style.use("bmh")
    plt.figure(figsize=(15,5))
    plt.title(f'{TICKER} Stock Price Forecasting')
    plt.xlabel("Days")
    plt.ylabel("Closing Price USD")
    #plt.plot(prices['close'], label='Closing Price')
    plt.plot(validation['close'], 'g', label='Original Price')
    plt.plot(validation['tree_pred'], 'r', label='Decision Tree Regressor')
    plt.plot(validation['knn_pred'], label='KNN Regressor')
    plt.plot(validation['el_pred'], label='ElasticNet Regressor')
    plt.plot(validation['ada_pred'], label='AdaBoost Regressor')
    plt.plot(validation['bay_pred'], label='BaysianRidge Regressor')
    plt.plot(validation['mlp_pred'], label='MLP Regressor')
    plt.plot(validation['lr_pred'], 'y', label='Linear Regression')
    plt.legend()
    plt.show()