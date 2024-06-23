from ma_crossover_strategy import MACrossoverStrategy
from optimization_mixin import OptimizationMixin
import numpy as np
import math
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SMACrossoverStrategy(MACrossoverStrategy, OptimizationMixin):
    def generate_signals(self) -> None:
        self.data['Short MA'] = self.data['Adj Close'].rolling(window=self.short_window).mean()
        self.data['Long MA'] = self.data['Adj Close'].rolling(window=self.long_window).mean()
                                         
        self.data['Signal'] = self.mode # 1 is buy, -1 is sell and start at mode
        self.data['Signal'][self.long_window-1:] = np.where(self.data['Short MA'][self.long_window-1:]>self.data['Long MA'][self.long_window-1:], 1, -1)
        self.data['Position'] = self.data['Signal'].diff()
        self.num_trades = math.ceil(self.data['Position'].abs().sum()/2)
    
    def optimize_parameters(self) -> None:
        #Train-test split
        train_data, test_data = self.train_test_split()
        self.best_parameters, self.best_metrics = self.optimize(self.bounds, train_data, test_data)

                                          
                                       
                                        
                                          
