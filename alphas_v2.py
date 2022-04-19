# IMPORTS
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

from asyncio.windows_events import NULL

import streamlit as st

import pandas as pd
import numpy as np

import datetime
import pandas_datareader as data
import yfinance as yf

from plotly import graph_objs as go
import plotly.express as px
import cufflinks as cf

from urllib.request import urlopen, Request
from bs4 import BeautifulSoup

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# PARAMETERS
START = "2012-01-01"
TODAY = datetime.date.today().strftime("%Y-%m-%d")
END = TODAY
NAME_DELIMETER = ' - '


def scrape_news(ticker):
# INPUT: 
    # tickers: list of tickers to scrape news for online
# OUTPUT:
    # news_tables: list of news html for each ticker
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    print('ticker','>'*50,ticker)
    stock_url = finviz_url + ticker

    req = Request(url=stock_url, headers={'user-agent': 'news_app'})
    print('-'*50, req)
    response = urlopen(req)
    html = BeautifulSoup(response, features='html')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    return news_tables


def parse_news(news_tables):
 
    parsed_news = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):

            news_title = row.a.text
            timestamp = row.td.text.split(' ')

            if len(timestamp) == 1:
                time = timestamp[0]
            else:
                date = timestamp[0]
                time = timestamp[1]
        
            parsed_news.append([ticker, date, time, news_title])
    return pd.DataFrame(parsed_news, columns=['ticker', 'date', 'time', 'news_title'])

def get_news_polarity(news_df):
    pf = lambda nt: vader.polarity_scores(nt)['compound']
    news_df['compound'] = round(news_df['news_title'].apply(pf),2)
    return news_df

def get_daily_sentiment(news_df):
    
    daily_sentiment = news_df.groupby(['ticker', 'date']).mean()
    daily_sentiment = daily_sentiment.unstack().xs('compound', axis='columns').transpose()

    return daily_sentiment 




def load_data(ticker, start_date, end_date):
# loads stock O H L C AdjC prices and Volume from Yahoo !
    # df = data.DataReader(ticker, 'yahoo', START, END)
    # df.reset_index(inplace=True)
    
    stock_data = yf.Ticker(ticker)
    df = stock_data.history(period='1d', start=start_date, end=end_date)
    # df.reset_index(inplace=True)
    # df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
    return stock_data, df

def plot_raw_data(stock):
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Adj Close'], name='stock_open'))
    # fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Volume'], name='stock_volume'))
    fig.add_trace(go.Scatter(x=stock['Date'], y=stock['Close'], name='stock_open'))    
    fig.layout.update(title_text="Time Sereies Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def draw_BB(stock_price, period):
    qf = cf.QuantFig(stock_price, legend='left')
    qf.add_bollinger_bands(periods=period)
    fig = qf.iplot(asFigure=True)
    st.plotly_chart(fig)

def display_logo(stock_data):
    if 'logo_url' in stock_data.info.keys():
        logo = '<img src=%s>' % stock_data.info['logo_url']
        if logo:
            st.markdown(logo, unsafe_allow_html=True)

def display_company_name(stock_data):
    if 'longName' in stock_data.info.keys():
        if stock_data.info['longName']:
            st.header('**%s**' % stock_data.info['longName'])

def display_business_summary(stock_data):
    if 'longBusinessSummary' in stock_data.info.keys():
        if stock_data.info['longBusinessSummary']:
            st.info(stock_data.info['longBusinessSummary'])  

def display_business_data(data):
    # Display Company Logo
    display_logo(data)

    # Display Company Logo
    display_company_name(data)

    # Display Business Summary 
    display_business_summary(data)


def get_pct(df, freq):
    freq = freq[0]
    if freq == 'D':
        ret = df.pct_change(periods=1, freq=freq) * 100
        ret = round(ret, 2) 
        ret.dropna(inplace=True)        
        st.line_chart(ret)
    else:
        ret = df.pct_change(periods=1, freq=freq) * 100
        ret = round(ret, 2) 

        st.bar_chart(ret)
    # cum_ret = (1+ret).cumprod() - 1
    # cum_ret.fillna(0, inplace=True)
    # return round(cum_ret,2)

def ticker_data(selected_stock, start_date, end_date):
    st.subheader('**Ticker Data**')
    stock_ohlc = round(yf.download(selected_stock, start_date, end_date),2)
    st.dataframe(stock_ohlc.sort_index(ascending=False).head())
    stock_price = stock_ohlc['Adj Close']
    st.line_chart(stock_price)   

scaler = MinMaxScaler(feature_range=(0,1))


st.sidebar.subheader('Query Parameters')
start_date = st.sidebar.date_input('Start Date', datetime.date(2015, 1,1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())

tickers_df = pd.read_csv("nasdaq_stocks.csv")
tickers = tickers_df['Symbol'].map(str) + ' - ' + tickers_df['Name'].map(str)
tickers.iloc[0] = 'None'
selected_stock = st.sidebar.selectbox('Select Company', tickers)


if selected_stock != 'None':
    selected_stock = selected_stock.split('-')[0]  
    print('\n\n\n\n', selected_stock)  
    # Load Business Data and Price Data
    stock_data, _ = load_data(selected_stock, start_date, end_date)

    # Display Business Data
    display_business_data(stock_data)


    # Display recent stock OHLC data and Price Chart
    ticker_data(selected_stock, start_date, end_date)

    st.write('---')

    st.subheader('**Comparison**')    
     
    compare_stocks = st.multiselect('Choose Stocks to Compare', tickers_df)
    pct_date_col1, pct_date_col2, freq_col3 = st.columns(3)    
    with pct_date_col1:
        pct_start_date = st.date_input('From', start_date)
    with pct_date_col2:        
        pct_end_date = st.date_input('To', end_date)
    with freq_col3:
        freq = ['Daily', 'Montly', 'Yearly']
        sel_freq = st.select_slider('Select Frequency', freq)
    st.caption('**Returns % Comparison**')   
    if len(compare_stocks) > 0:
        comp_stock_price = yf.download(compare_stocks, pct_start_date, pct_end_date)['Adj Close']
        get_pct(comp_stock_price, sel_freq)

    st.caption('**Volume % Comparison**')    
    if len(compare_stocks) > 0:
        comp_stock_vol = yf.download(compare_stocks, pct_start_date, pct_end_date)['Volume']
        pct_vol = get_pct(comp_stock_vol, sel_freq)      
    st.write('---')

    st.subheader('**New Sentiment Analysis**')

    
    vader = SentimentIntensityAnalyzer()
    print('v'*50,selected_stock)
    news_tables = scrape_news(selected_stock)
    news_df = parse_news(news_tables)
    news_pol_df = get_news_polarity(news_df)

    news_pol_df['timestamp'] = news_pol_df['date'] +"|"+ news_pol_df['time']
    fig = px.bar(
                news_pol_df, 
                x = 'timestamp', 
                y='compound',
                color = "compound",
                template = 'seaborn', 
                color_continuous_scale=px.colors.diverging.RdYlGn
                )
    graph_title = "News Sentiment Graph for " + stock_data.info['longName']
    fig.update_layout(
                    title_x=0,
                    margin= dict(l=0,r=10,b=10,t=30),
                    yaxis_title=None,
                    xaxis_title=None,
                    legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=0.9,
                                xanchor="right",
                                x=0.99
                                )
                    )                
    st.plotly_chart(fig, use_container_width=True) 

    st.write('---')
    st.subheader('**10 Sentiment Analysis**')        
    st.write('---')
    st.subheader('**Portfolio Optimization**')        
    pf_col1, pf_col2 = st.columns(2)    
    with pf_col1:
        portfolio_stocks = st.multiselect('Select Stocks for your portfolio', tickers_df)
    with pf_col2:
        sel_amount = st.slider('Investment Amount', 20.00, 5000.00,step=1.00)
    

    stock_prices = pd.DataFrame()

    for ticker in portfolio_stocks:
        stock_prices[ticker] = yf.download(ticker, start = start_date, end = end_date)['Adj Close']    

    if st.button('Optimize Portfolio'):
        # daily percentage returns
        daily_returns = stock_prices.pct_change()

        # annualize returns covariance matrix
        cov_matrix_annual = daily_returns.cov() * 252
        
        # portfolio variance
        weights = np.array([1/len(portfolio_stocks)]*len(portfolio_stocks)) 
        pf_var = np.dot(weights.T, np.dot(cov_matrix_annual, weights))    

        # portfolio volatility
        pf_vol = np.sqrt(pf_var)

        # annual returns £
        pf_annual_return = np.sum(daily_returns.mean() * weights) * 252
        st.subheader('Baseline Portfolio Metrics')
        st.metric('Annual Return', f"{pf_annual_return*100:.2f}%")
        st.metric('Annual Volatility', f"{pf_vol*100:.2f}%")


        # portfolio optimization happens here
        mu = expected_returns.mean_historical_return(stock_prices)
        S = risk_models.sample_cov(stock_prices)
        ef = EfficientFrontier(mu, S)
    
        # get the most effiecient weights maximizing the sharpe ratio
        weights = ef.max_sharpe()
        clean_weights = ef.clean_weights()

        # compare optimized return/risk with equal weighted portfolio return/risk in beginning
        ef_ret, ef_vol, _ = ef.portfolio_performance(verbose=True)    
        st.subheader('Optimized Portfolio Metrics')
        st.metric('Annual Return', f"{ef_ret*100:.2f}%")
        st.metric('Annual Volatility', f"{ef_vol*100:.2f}%")

        # get share allocation based on money to be invested
        latest_prices = get_latest_prices(stock_prices)

        row = []
        for ticker, weight in  clean_weights.items():
            num_shares = int((sel_amount*weight)/stock_prices[ticker][-1])
            allocation = stock_prices[ticker][-1] * num_shares
            row.append([ticker, round(weight,2), round(stock_prices[ticker][-1],2), num_shares, round(allocation,2)])

        fund_allocation = pd.DataFrame(row, columns=['ticker', 'weight', 'price', 'num_shares', 'allocation'])
        print(fund_allocation)
        st.caption("Allocation")
        COMMON_ARGS = {
        "color": "allocation",
        "color_discrete_sequence": px.colors.sequential.Greens,
        "hover_data": [
            "ticker",
            "price",
            "num_shares",
            "allocation",
        ],
        }        
        fig = px.pie(fund_allocation, values="allocation", names="ticker", hover_data=['ticker', 'price', 'num_shares', 'allocation'])
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig)     


        



        
    st.write('---')