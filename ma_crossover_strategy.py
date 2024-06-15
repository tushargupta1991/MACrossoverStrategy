import yfinance as yf
import pandas as pd
import numpy as np 
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
from typing import Tuple
import matplotlib.pyplot as plt
from config import FILE_SAVE_DIRECTORY
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class MACrossoverStrategy(ABC):
    symbol: str = 'AAPL'
    short_window: int = 5
    long_window: int = 20
    start: str = '2010-01-01'
    end: str = field(default_factory=lambda: (datetime.datetime.today()-datetime.timedelta(days=1)).strftime('%Y-%m-%d'))
    mode: int = 1 # 1 => long only trades, -1 => short only trades
    plot: bool = False
    bounds: Tuple = field(default_factory=lambda: ((3, 30), (20, 200)))
    train_split_ratio: float = 0.8
    num_trades: int = field(init=False, default=0)
    data: pd.DataFrame = field(init=False, default=None)
    metrics: dict = field(init=False, default=None)

    def fetch_data(self) -> None:
        self.data = yf.download(self.symbol, start=self.start, end=self.end)
        if self.data.empty:
            raise ValueError('Data not available for the given dates.')
        logger.info(f'Data fetched for {self.symbol} from {self.start} to {self.end}: {self.data.head()}')
    
    @abstractmethod
    def generate_signals(self) -> None:
        pass

    def calculate_performance(self) -> Tuple:
        #Calculate returns
        self.data['Market Return'] = self.data['Adj Close'].pct_change()
        if self.mode == 0: #No preference for long or short trades
            self.data['Strategy Return'] = self.data['Market Return']*self.data['Signal'].shift(1)
        else:
            self.data['Strategy Return'] = self.data['Market Return']*self.mode*(self.data['Signal'].shift(1) == self.mode).astype(int)

        #Calculate cumulative returns
        self.data['Cumulative Strategy Return'] = (1 + self.data['Strategy Return']).cumprod() - 1
        self.data['Cumulative Market Return'] = (1 + self.data['Market Return']).cumprod() - 1

        #Calculate drawdown
        self.data['Cumulative Strategy High'] = self.data['Cumulative Strategy Return'].cummax()
        self.data['Strategy Drawdown'] = self.data['Cumulative Strategy High'] - self.data['Cumulative Strategy Return']
        self.data['Cumulative Market High'] = self.data['Cumulative Market Return'].cummax()
        self.data['Market Drawdown'] = self.data['Cumulative Market High'] - self.data['Cumulative Market Return']

        #Calculate Sharpe Ratio
        sharpe_ratio_strategy = self.data['Strategy Return'].mean()/self.data['Strategy Return'].std()
        sharpe_ratio_strategy_annualized = sharpe_ratio_strategy * np.sqrt(252)
        sharpe_ratio_market = self.data['Market Return'].mean()/self.data['Market Return'].std()
        sharpe_ratio_market_annualized = sharpe_ratio_market * np.sqrt(252)

        self.sharpe_ratio_strategy_annualized = sharpe_ratio_strategy_annualized
        logger.info(f'Strategy sharpe ratio: {sharpe_ratio_strategy_annualized=} vs market sharpe ratio: {sharpe_ratio_market_annualized=}')

        #Summarize all the metrics
        self.metrics = {'Sharpe Ratio': [round(sharpe_ratio_strategy_annualized,2), round(sharpe_ratio_market_annualized,2)], 
                        'Drawdown': [f'{100*self.data["Strategy Drawdown"].max():.2f}%', f'{100*self.data["Market Drawdown"].max():.2f}%'],
                        'Total Return': [f"{100*self.data['Cumulative Strategy Return'].iloc[-1]:.2f}%", f"{100*self.data['Cumulative Market Return'].iloc[-1]:.2f}%"], 
                        'Volatility': [f"{100*self.data['Strategy Return'].std()*np.sqrt(252):.2f}%", f"{100*self.data['Market Return'].std()*np.sqrt(252):.2f}%"],
                        'Drawdown': [f'{100*self.data["Strategy Drawdown"].max():.2f}%', f'{100*self.data["Market Drawdown"].max():.2f}%'],
                        'Cumulative High': [f"{100*self.data['Cumulative Strategy High'].iloc[-1]:.2f}%", f"{100*self.data['Cumulative Market High'].iloc[-1]:.2f}%"],
                        'Total Trades': [self.num_trades, 'N/A']}
        self.metrics = pd.DataFrame(self.metrics, index=['Strategy', 'Market']).T

    def plot_performance(self) -> None:
        #Plotting and saving the data
        if self.data is not None and self.plot:
            plt.clf()
            plt.plot(100*self.data['Cumulative Strategy Return'], label='Strategy')
            plt.plot(100*self.data['Cumulative Market Return'], label='Market')
            plt.legend()
            plt.xlabel('Date')
            plt.ylabel('Return (%)')
            plt.title('Cumulative Returns')
            
            '''
            # Convert the DataFrame to a 2D list for the table function
            table_data = self.metrics.values.tolist()

            # Add a table at the bottom of the axes
            the_table = plt.table(cellText=table_data, colLabels=self.metrics.columns, rowLabels=self.metrics.index, loc='bottom', cellLoc='center',bbox=[0, -0.5, 1, 0.5])
            # Adjust table properties for better presentation
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 1.5)
            # Adjust layout to make room for the table
            plt.subplots_adjust(left=0.1, bottom=0.3, top=0.9, right=0.9)
            '''

            plot_directory = os.path.join(FILE_SAVE_DIRECTORY, 'Plots')
            os.makedirs(plot_directory, exist_ok=True)
            plt.savefig(os.path.join(plot_directory, f'cumulative_returns_{self.symbol}_{self.start}_{self.end}_{self.short_window}_{self.long_window}_{self.mode}.png'))

            #Saving plot csv
            plot_directory = os.path.join(FILE_SAVE_DIRECTORY, 'Data')
            os.makedirs(plot_directory, exist_ok=True)
            self.data.to_csv(os.path.join(plot_directory, f'cumulative_returns_{self.symbol}_{self.start}_{self.end}_{self.short_window}_{self.long_window}_{self.mode}.csv'))
    
    def train_test_split(self) -> Tuple:
        split_index = int(self.train_split_ratio*len(self.data))
        train_data = self.data[:split_index]
        test_data = self.data[split_index:]
        return train_data, test_data

    def run(self) -> Tuple:
        if self.data is None or self.data.empty: self.fetch_data()
        self.generate_signals()
        self.calculate_performance()
        self.plot_performance()

if __name__ == '__main__':
    data = yf.download('AAPL', start='2010-01-01', end='2020-01-01')
    print(data['Adj Close'])