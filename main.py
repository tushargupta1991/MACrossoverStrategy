from ma_strategy import MACrossoverStrategy
from tsmc_adr_strategy import TSMCLongShortStrategy
from config import FILE_SAVE_DIRECTORY
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    '''
    #MACrossover Strategy
    sma = MACrossoverStrategy(symbol='TSLA', start='2010-01-01',short_window=5, long_window=20, mode=-1, plot=True)
    sma.run()
    print(sma.metrics)   

    sma.optimize_parameters()
    print(sma.best_params)
    print(sma.train_metrics)
    print(sma.test_metrics)
    '''

    #TSMC Long Short Strategy
    tsmc = TSMCLongShortStrategy(symbol='TSMC', symbol_taiwan='2330.TW', symbol_adr='TSM', start='2020-01-01', plot=True)
    tsmc.run(neutral=True)
    print(tsmc.metrics)

if __name__ == '__main__':
    main()