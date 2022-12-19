import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt 
from yahoofinancials import YahooFinancials

START = '2018-01-01'
END = '2022-11-30'
TICKER = 'AAPL'

def get_data(ticker, start_date, end_date):
    yahoo_financials = YahooFinancials(ticker)
    price_data = yahoo_financials.get_historical_price_data(start_date, end_date, 'daily')
    df = pd.DataFrame(price_data[ticker]['prices'])
    df.drop(['date', 'adjclose', 'formatted_date', 'high', 'low', 'open', 'volume'], axis=1, inplace=True)
    return df

def plot_ma(ticker, train_df):
    ma100 = train_df['close'].rolling(100).mean()
    ma200 = train_df['close'].rolling(200).mean()
    plt.figure(figsize = (15,5))
    plt.plot(train_df['close'], label='Training Prices')
    plt.plot(ma100, 'r', label='100 Day Moving Average')
    plt.plot(ma200, 'g', label='200 Day Moving Average')
    plt.title(f'{ticker} Price Chart')
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(loc='upper left')
    plt.show()  


if __name__ == "__main__":
    prices = get_data(TICKER, START, END)
    #plot_ma(TICKER, prices)

    future_days = 30
    prices['prediction'] = prices[['close']].shift(-future_days)
    x = np.array(prices['close'])[:-future_days].reshape(-1,1)
    y = np.array(prices['prediction'])[:-future_days].reshape(-1,1)

    X_train, X_test ,y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
    tree = DecisionTreeRegressor().fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)
    knn = KNeighborsRegressor().fit(X_train, y_train)

    x_future = prices['close'][:-future_days]
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future).reshape(-1,1)

    tree_pred = tree.predict(x_future)
    lr_pred = lr.predict(x_future)
    knn_pred = lr.predict(x_future)

    validation = prices[['close']][x.shape[0]:]
    validation['tree_pred'] = tree_pred
    validation['lr_pred'] = lr_pred
    validation['knn_pred'] = knn_pred
    
    plt.style.use("bmh")
    plt.figure(figsize=(15,5))
    plt.title(f'{TICKER} Stock Price Forecasting')
    plt.xlabel("Days")
    plt.ylabel("Closing Price USD")
    plt.plot(prices['close'], label='Closing Price')
    #plt.plot(validation['close'], 'g', label='Original Price')
    plt.plot(validation['tree_pred'], 'r', label='Decision Tree Regressor')
    #plt.plot(validation['knn_pred'], label='KNN Regressor')
    #plt.plot(validation['lr_pred'], 'y', label='Linear Regression')
    plt.legend()
    plt.show()