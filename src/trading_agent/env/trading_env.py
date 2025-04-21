from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

class TradingEnvBuilder:
    """
    Class for building trading environments
    """
    @staticmethod
    def create_env(data, config):
        """
        Create a stock trading environment
        
        Args:
            data (DataFrame): Processed data for the environment
            config (Config): Configuration settings
            
        Returns:
            StockTradingEnv: Trading environment
        """
        # Calculate state space size
        stock_dimension = len(data.tic.unique())
        state_space = 1 + 2*stock_dimension + len(config.INDICATORS)*stock_dimension
        print(data.head())
        
        # Environment configuration
        env_config = {
            "df": data,
            "stock_dim": stock_dimension,
            "hmax": config.MAX_SHARES_PER_TRADE,
            "initial_amount": config.INITIAL_AMOUNT,
            "reward_scaling": config.REWARD_SCALING,
            "state_space": state_space,
            "action_space": stock_dimension,
            "tech_indicator_list": config.INDICATORS,
            "turbulence_threshold": config.TURBULENCE_THRESHOLD,
            "buy_cost_pct": config.BUY_COST_PCT,
            "sell_cost_pct": config.SELL_COST_PCT,
            "num_stock_shares": [0] * stock_dimension,
        }
        
        # Create environment
        env = StockTradingEnv(**env_config)
        
        return env
    