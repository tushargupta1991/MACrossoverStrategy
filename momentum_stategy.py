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
class MomentumStrategy(StrategyTemplate):
    momentum_window: int = 20
    upper_threshold: float = 0.05
    lower_threshold: float = -0.05
    mode: int = 0
    bounds: Tuple = field(default_factory=lambda: ((5,30), (-0.1, -0.01), (0.01, 0.1)))

    def generate_signals(self) -> None:
        if self.data is None or self.data.empty:
            self.fetch_data()
        
        #Calculate momentum (rate of change)
        self.data['Momentum'] = self.data['Adj Close'].pct_change(periods=self.momentum_window)

        #Generating signals
        self.data.loc[self.data.index[self.momentum_window-1:],'Signal'] = np.where(self.data.loc[self.data.index[self.momentum_window-1:],'Momentum']>self.upper_threshold, 1, np.where(self.data.loc[self.data.index[self.momentum_window-1:],'Momentum']<self.lower_threshold, -1, 0))

        self.data['Position'] = self.data['Signal'].diff()

        #Getting num trades
        self.num_trades = math.ceil(self.data['Position'].abs().sum()/2)
    
    def optimize_parameters(self) -> None:
        self.plot = False

        train_data, test_data = self.train_test_split()
        
        # Define the objective function
        def objective_function(params):
            momentum_window, lower_threshold, upper_threshold = params
            self.momentum_window = int(momentum_window)
            self.upper_threshold = upper_threshold
            self.lower_threshold = lower_threshold
            self.data = train_data.copy()
            self.run()
            return -self.sharpe_ratio_strategy_annualized
        
        # Perform the optimization
        result = differential_evolution(objective_function, self.bounds)
        best_params = list(result.x)
        best_params[0] = int(best_params[0])
        self.best_params = best_params
        logger.info('Optimal parameters: %s', self.best_params)

        isPlot = self.plot 

        #Evaluate on the train data
        self.data = train_data.copy()
        self.plot = 'train'
        self.momentum_window, self.lower_threshold, self.upper_threshold = self.best_params
        self.run()
        self.train_metrics = self.metrics
        logger.info('Metrics on the train data: %s', self.metrics)

        #Evaluate on the test data
        self.data = test_data.copy()
        self.plot = 'test'
        self.momentum_window, self.lower_threshold, self.upper_threshold = self.best_params
        self.run()
        self.test_metrics = self.metrics
        logger.info('Metrics on the test data: %s', self.metrics)

        #Restoring the value
        self.plot = isPlot