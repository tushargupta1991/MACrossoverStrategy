from strategy_template import StrategyTemplate
import numpy as np
import pandas as pd
import math, os
from dataclasses import dataclass, field
import logging
from scipy.optimize import differential_evolution
from typing import Tuple
import matplotlib.pyplot as plt
from config import FILE_SAVE_DIRECTORY
import yfinance as yf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class TSMCLongShortStrategy(StrategyTemplate):
    symbol_taiwan: str = '2330.TW'
    symbol_adr: str = 'TSM'
    window: int = 20

    def fetch_data(self):
        # Fetch data for both TSMC Taiwan and ADR
        data_taiwan = yf.download(self.symbol_taiwan, start=self.start, end=self.end)
        data_adr = yf.download(self.symbol_adr, start=self.start, end=self.end)
        currency_conversion = yf.download('TWD=X', start=self.start, end=self.end)

        if data_taiwan.empty or data_adr.empty or currency_conversion.empty:
            raise ValueError("Data not available for one or more symbols.")

        # Merge the data on the date index
        data_taiwan = data_taiwan['Close']
        data_adr = data_adr['Close']
        currency_conversion = currency_conversion['Close']

        data = pd.merge(data_taiwan, data_adr, left_index=True, right_index=True, suffixes=('_taiwan', '_adr'))
        data = pd.merge(data, currency_conversion, left_index=True, right_index=True)

        # Adjust the TSMC Taiwan stock price to USD
        data['Close_taiwan_usd'] = data['Close_taiwan'] / data['Close']

        data.rename(columns={'Close':'Close currency','Close_taiwan':'Close Taiwan (in TWD)','Close_taiwan_usd': 'Close Taiwan','Close_adr': 'Close ADR'}, inplace=True)

        self.data = data

    def generate_signals(self) -> None:
        if self.data is None or self.data.empty:
            self.fetch_data()
        
        #Calculate the spread in USD
        self.data['Spread'] = self.data['Close ADR'] - self.data['Close Taiwan'] #in USD

        #Calculate the MA
        self.data['Spread MA'] = self.data['Spread'].rolling(window=self.window).mean()
        self.data['Spread STD'] = self.data['Spread'].rolling(window=self.window).std()

        #Generating signals
        self.data['Signal'] = 0
        self.data.loc[self.data['Spread'] > self.data['Spread MA']+self.data['Spread STD'],'Signal'] = -1
        self.data.loc[self.data['Spread'] < self.data['Spread MA']+self.data['Spread STD'],'Signal'] = 1

        self.data['Position'] = self.data['Signal'].diff()

        #Getting num trades
        self.num_trades = math.ceil(self.data['Position'].abs().sum()/2)