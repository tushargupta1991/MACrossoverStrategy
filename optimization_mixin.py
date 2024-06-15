from scipy.optimize import differential_evolution
import logging

logger = logging.getLogger(__name__)

class OptimizationMixin:
    def optimize(self, bounds, train_data, test_data):
        def objective(params):
            params = [int(p) for p in params]
            self.data = train_data.copy()
            print(params)
            self.short_window, self.long_window = params
            self.run()
            print(self.sharpe_ratio_strategy_annualized)
            return self.sharpe_ratio_strategy_annualized * -1

        result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=100,  disp=True)
        best_params = [int(p) for p in result.x]

        #Evaluate on the test data
        self.data = test_data.copy()
        
        logger.info(f'Optimal parameters: {best_params}')

        self.short_window, self.long_window = best_params
        self.run()
        return best_params, self.metrics