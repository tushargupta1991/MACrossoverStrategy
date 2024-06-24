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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@dataclass
class MACrossoverStrategy(StrategyTemplate):
    short_window: int = 5
    long_window: int = 20
    mode: int = 1 # 1 => long only trades, -1 => short only trades
    bounds: Tuple = field(default_factory=lambda: ((3, 30), (20, 200)))
    best_params: Tuple = field(init=False, default=None)
    train_metrics: pd.DataFrame = field(init=False, default=None)
    test_metrics: pd.DataFrame = field(init=False, default=None)
    
    def generate_signals(self) -> None:
        self.data['Short MA'] = self.data['Adj Close'].rolling(window=self.short_window).mean()
        self.data['Long MA'] = self.data['Adj Close'].rolling(window=self.long_window).mean()
                                         
        self.data['Signal'] = 0
        self.data.loc[self.data.index[self.long_window-1:],'Signal'] = np.where(self.data.loc[self.data.index[self.long_window-1:],'Short MA']>self.data.loc[self.data.index[self.long_window-1:],'Long MA'], 1, -1)
        self.data['Position'] = self.data['Signal'].diff()

        #Getting num trades
        self.num_trades = math.ceil(self.data['Position'].abs().sum()/2)

        # Adding a column for signal changes
        self.data['Signal_Change'] = self.data['Signal'].diff()
    
    def plot_performance(self) -> None:
        if self.data is not None and self.plot:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14))
        
            # First plot: SMAs and Close Price
            ax1.plot(self.data['Short MA'], label=f'SMA_{self.short_window}')
            ax1.plot(self.data['Long MA'], label=f'SMA_{self.long_window}')
            ax1.plot(self.data['Adj Close'], label='Close Price', color='black', linestyle='--')

            # Add vertical bars for signal changes, skipping the first signal change
            first_signal_change = False
            for i in range(1, len(self.data)):
                if self.data['Signal_Change'].iloc[i] != 0:
                    if first_signal_change:
                        first_signal_change = False
                        continue
                    color = 'green' if self.data['Signal_Change'].iloc[i] > 0 else 'red'
                    ax1.axvline(x=self.data.index[i], color=color, linestyle='--', linewidth=1)
            ax1.legend()
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price')
            ax1.set_title('SMAs and Close Price')
            
            # Second plot: Cumulative Returns
            ax2.plot(100 * self.data['Cumulative Strategy Return'], label='Cumulative Strategy Return')
            ax2.plot(100 * self.data['Cumulative Market Return'], label='Cumulative Market Return')
            
            # Add vertical bars for signal changes, skipping the first signal change
            first_signal_change = True
            for i in range(1, len(self.data)):
                if self.data['Signal_Change'].iloc[i] != 0:
                    if first_signal_change:
                        first_signal_change = False
                        continue
                    color = 'green' if self.data['Signal_Change'].iloc[i] > 0 else 'red'
                    ax2.axvline(x=self.data.index[i], color=color, linestyle='--', linewidth=1)
            
            ax2.legend()
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Return (%)')
            ax2.set_title('Cumulative Returns')
            
            # Convert the DataFrame to a 2D list for the table function
            table_data = self.metrics.values.tolist()
            
            # Add a table at the bottom of the second plot
            the_table = ax2.table(cellText=table_data, colLabels=self.metrics.columns, rowLabels=self.metrics.index, loc='bottom', cellLoc='center', bbox=[0, -0.5, 1, 0.3])
            # Adjust table properties for better presentation
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(10)
            the_table.scale(1, 1.5)
            # Adjust layout to make room for the table
            plt.subplots_adjust(left=0.1, bottom=0.2, top=0.9, right=0.9, hspace=0.5)

            traintest = '' if self.plot not in ('train', 'test') else f'_{self.plot}'
            #Saving plot
            plot_directory = os.path.join(FILE_SAVE_DIRECTORY, 'Plots')
            os.makedirs(plot_directory, exist_ok=True)
            plt.savefig(os.path.join(plot_directory, f'cumulative_returns_{self.symbol}_{self.start}_{self.end}_{self.short_window}_{self.long_window}_{self.mode}{traintest}.png'))
            logger.info('Plot saved successfully.')

            #Saving plot csv
            plot_directory = os.path.join(FILE_SAVE_DIRECTORY, 'Data')
            os.makedirs(plot_directory, exist_ok=True)
            self.data.to_csv(os.path.join(plot_directory, f'cumulative_returns_{self.symbol}_{self.start}_{self.end}_{self.short_window}_{self.long_window}_{self.mode}{traintest}.csv'))
            logger.info('Data saved successfully.')

    def optimize_parameters(self) -> None:
        self.plot = False

        #Train-test split
        train_data, test_data = self.train_test_split()

        def objective(params):
            self.short_window, self.long_window = int(params[0]), int(params[1])
            self.data = train_data.copy()
            if self.short_window >= self.long_window:
                return np.inf  # Invalid parameter combination
            self.run()
            return -self.sharpe_ratio_strategy_annualized

        result = differential_evolution(objective, self.bounds, strategy='best1bin', maxiter=100,  disp=True)
        best_params = [int(p) for p in result.x]
        self.best_params = best_params
        logger.info('Optimal parameters: %s', best_params)

        isPlot = self.plot 

        #Evaluate on the train data
        self.data = train_data.copy()
        self.plot = 'train'
        self.short_window, self.long_window = best_params
        self.run()
        self.train_metrics = self.metrics
        logger.info('Metrics on the train data: %s', self.metrics)

        #Evaluate on the test data
        self.data = test_data.copy()
        self.plot = 'test'
        self.short_window, self.long_window = best_params
        self.run()
        self.test_metrics = self.metrics
        logger.info('Metrics on the test data: %s', self.metrics)

        #Restoring the value
        self.plot = isPlot

                                          
                                       
                                        
                                          
