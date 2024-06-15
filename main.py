from sma_strategy import SMACrossoverStrategy
from config import FILE_SAVE_DIRECTORY
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def main():
    sma = SMACrossoverStrategy('TSLA', 3, 30, mode=1)
    sma.run()
    print(sma.metrics)   
    # sma.optimize_parameters() 
    # print(sma.best_parameters)
    # print(sma.best_metrics)

if __name__ == '__main__':
    main()