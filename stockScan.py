import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fetch stock data
def fetch_stock_data(ticker):
    try:
        stock_data = yf.download(ticker, period="1y", interval="1d")
        stock_data['Daily_Return'] = stock_data['Close'].pct_change()
        stock_data.dropna(inplace=True)  # Drop any rows with NaN values for smoother analysis
        return stock_data
    except Exception as e:
        logging.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

# Calculate technical indicators: RSI (Wilder's method), MACD, Moving Averages
def calculate_indicators(stock_data, short_window=40, long_window=100):
    # Moving Averages
    stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window).mean()

    # RSI Calculation using Wilder's method
    delta = stock_data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    stock_data['RSI'] = 100 - (100 / (1 + rs))

    # MACD Calculation
    stock_data['12_EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['26_EMA'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = stock_data['12_EMA'] - stock_data['26_EMA']
    stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()

    stock_data.dropna(inplace=True)
    return stock_data

# Set buy and sell signals based on indicators
def set_trading_signals(stock_data):
    stock_data['Buy_Signal'] = np.where(
        (stock_data['Short_MA'] > stock_data['Long_MA']) &
        (stock_data['RSI'] < 30) &
        (stock_data['MACD'] > stock_data['Signal_Line']), 1, 0
    )
    stock_data['Sell_Signal'] = np.where(
        (stock_data['Short_MA'] < stock_data['Long_MA']) |
        (stock_data['RSI'] > 70) |
        (stock_data['MACD'] < stock_data['Signal_Line']), -1, 0
    )
    return stock_data

# Simulate trades based on profit target and stop loss
def simulate_trades(stock_data, profit_target=0.10, stop_loss=0.05):
    initial_capital = 10000
    capital = initial_capital
    position = 0
    entry_price = 0
    trade_log = []

    for i in range(1, len(stock_data)):
        if position == 0 and stock_data['Buy_Signal'].iloc[i-1] == 1:
            position = capital / stock_data['Close'].iloc[i]
            entry_price = stock_data['Close'].iloc[i]
            capital = 0  # Invest all capital
            logging.info(f"Bought {position:.2f} shares at {entry_price:.2f} on {stock_data.index[i].date()}")
        elif position > 0:
            current_price = stock_data['Close'].iloc[i]
            profit = (current_price - entry_price) / entry_price
            if profit >= profit_target or profit <= -stop_loss or stock_data['Sell_Signal'].iloc[i-1] == -1:
                capital = position * current_price
                logging.info(f"Sold at {current_price:.2f} on {stock_data.index[i].date()} with profit: {profit:.2%}")
                position = 0
                entry_price = 0
                trade_log.append({'Date': stock_data.index[i], 'Capital': capital, 'Profit': profit})

    if position > 0:
        # Sell any remaining position at the last available price
        current_price = stock_data['Close'].iloc[-1]
        profit = (current_price - entry_price) / entry_price
        capital = position * current_price
        logging.info(f"Sold remaining position at {current_price:.2f} on {stock_data.index[-1].date()} with profit: {profit:.2%}")
        position = 0
        trade_log.append({'Date': stock_data.index[-1], 'Capital': capital, 'Profit': profit})

    total_profit = capital - initial_capital
    logging.info(f"Final capital: {capital:.2f}, Total profit: {total_profit:.2f}")
    return capital, trade_log

# Scan stocks and recommend buy candidates for the next day
def scan_and_recommend(tickers):
    recommendations = []
    for ticker in tickers:
        logging.info(f"Analyzing {ticker}...")
        stock_data = fetch_stock_data(ticker)
        if stock_data.empty:
            continue

        stock_data = calculate_indicators(stock_data)
        stock_data = set_trading_signals(stock_data)

        # Ensure we have enough data points
        if len(stock_data) < 2:
            continue

        # Check if today's signal is a buy recommendation for tomorrow
        if stock_data['Buy_Signal'].iloc[-1] == 1:
            recommendations.append(ticker)
            logging.info(f"{ticker} is a BUY candidate for tomorrow based on the analysis.")
        else:
            logging.info(f"{ticker} is NOT a good buy for tomorrow.")

    return recommendations

# Get NASDAQ-100 tickers
def get_nasdaq_100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/NASDAQ-100'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[1].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        logging.error(f"Error fetching NASDAQ-100 tickers: {e}")
        return []

# Get Dow Jones Industrial Average tickers
def get_dow_jones_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'class': 'wikitable sortable'})
        tickers = []
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[2].text.strip()
            tickers.append(ticker)
        return tickers
    except Exception as e:
        logging.error(f"Error fetching Dow Jones tickers: {e}")
        return []

# Main function to scan and recommend stocks
def main():
    nasdaq_tickers = get_nasdaq_100_tickers()
    dow_jones_tickers = get_dow_jones_tickers()

    if not nasdaq_tickers or not dow_jones_tickers:
        logging.error("Failed to retrieve tickers. Exiting the program.")
        return

    # Combine tickers from both indices
    all_tickers = list(set(nasdaq_tickers + dow_jones_tickers))

    # Limit to a manageable number of tickers for demonstration
    all_tickers = all_tickers[:50]

    # Scan and recommend stocks for the next day
    buy_recommendations = scan_and_recommend(all_tickers)

    # Output the recommended stocks for the next day
    if buy_recommendations:
        print("Stocks recommended to BUY tomorrow based on the analysis:")
        for stock in buy_recommendations:
            print(f"- {stock}")
    else:
        print("No stocks meet the buy criteria for tomorrow.")

if __name__ == "__main__":
    main()
