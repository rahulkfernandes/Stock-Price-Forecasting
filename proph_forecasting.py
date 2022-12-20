import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date
from yahoofinancials import YahooFinancials
from matplotlib import pyplot as plt

TICKER = 'AAPL'
START = '2012-01-01'
TODAY = date.today().strftime('%Y-%m-%d')
N_YEARS = 1  #Years of predictions


def get_data(ticker, start_date, end_date):
    yahoo_financials = YahooFinancials(ticker)
    price_data = yahoo_financials.get_historical_price_data(start_date, end_date, 'daily')
    df = pd.DataFrame(price_data[ticker]['prices'])
    df.drop(['adjclose', 'date', 'high', 'low', 'open','volume'], axis=1, inplace=True)
    return df

def plot_price(ticker, df):
    plt.figure(figsize = (15,5))
    plt.plot(df['close'], label='Prices')
    plt.title(f'{ticker} Closing Price Chart')
    plt.ylabel('CLosing Price')
    plt.xlabel('Days')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":

    data = get_data(TICKER, START, TODAY)
    #plot_price(TICKER, data)

    df_train = data[['formatted_date', 'close']]
    df_train = df_train.rename(columns={'formatted_date': 'ds', 'close': 'y'})

    period = N_YEARS * 365
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # fig1 = plot_plotly(m, forecast) ## For Plotly
    # fig1.show() 
    fig1 = m.plot(forecast)
    plt.ylabel('Closing Price')
    plt.xlabel('Days')
    plt.legend(loc='upper left')
    plt.show()

    fig2 = m.plot_components(forecast)
    plt.show()