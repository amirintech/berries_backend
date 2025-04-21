# Import modules
import os
import pandas as pd
from datetime import datetime
from config import Config
from data.data_fetcher import DataFetcher
from data.data_processor import DataProcessor
from env.trading_env import TradingEnvBuilder
from models.ppo_agent import PPOAgentTrainer
from evaluation.backtest import BacktestEvaluator
from evaluation.benchmarks import BenchmarkCalculator
from visualization.plotting import PerformancePlotter

class SPXTrader:
    """
    Main class for SPX trading with PPO agent
    """
    def __init__(self, api_key=None, secret_key=None, config=None):
        """
        Initialize the SPX trader
        
        Args:
            api_key (str): Alpaca API key
            secret_key (str): Alpaca API secret key
            config (Config): Configuration object
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.config = config or Config()
        
        # Initialize components
        self.data_fetcher = DataFetcher(api_key, secret_key)
        self.data_processor = DataProcessor(indicators=self.config.INDICATORS)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        
        # Environments
        self.train_env = None
        self.test_env = None
        
        # Model
        self.trainer = None
        self.model = None
        
        # Performance data
        self.account_value = None
        self.actions = None
        self.benchmarks = None
        self.performance_metrics = None
        
        # Create output directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
    
    def fetch_and_process_data(self):
        """
        Fetch and process data for training and testing
        """
        print("\n=== Step 1: Fetching and Processing Data ===")
        
        try:
            # Fetch data
            print(f"Fetching data for {self.config.TICKER}...")
            df = self.data_fetcher.fetch_data(
                self.config.TICKER,
                self.config.TRAIN_START_DATE,
                self.config.TEST_END_DATE
            )
            
            print(f"Raw data shape: {df.shape}")
            print(f"Raw data columns: {df.columns.tolist()}")
            
            # Check if data has multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                print("Data has multi-level columns")
                print(f"Column levels: {df.columns.names}")
                
                date = df.xs('date', axis=1, level='Price')
                gspc = df.xs('^GSPC', axis=1, level='Ticker')
                self.raw_data = pd.concat([date, gspc], axis=1)
                self.raw_data.columns = ['date', *gspc.columns.tolist()]
                self.raw_data['tic'] = 'SPX'
            else:
                print("Data does not have multi-level columns, using as is")
                self.raw_data = df
            
            print(f"Processed raw data shape: {self.raw_data.shape}")
            print(f"Processed raw data columns: {self.raw_data.columns.tolist()}")
            
            # Process data
            print("Adding technical indicators...")
            self.processed_data = self.data_processor.preprocess(self.raw_data)
            
            print(f"Processed data shape: {self.processed_data.shape}")
            print(f"Processed data columns: {self.processed_data.columns.tolist()}")
            
            # Split data
            print("Splitting data...")
            self.train_data, self.test_data = self.data_processor.split_data(
                self.processed_data,
                self.config.TRAIN_START_DATE,
                self.config.TRAIN_END_DATE,
                self.config.TEST_START_DATE,
                self.config.TEST_END_DATE
            )
            
            return self.processed_data
        except Exception as e:
            print(f"Error in data processing: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_environments(self):
        """
        Set up training and testing environments
        """
        print("\n=== Step 2: Setting Up Environments ===")
        
        try:
            # Print data shape before environment creation
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Training data columns: {self.train_data.columns.tolist()}")
            
            # Create environments
            print("Creating training environment...")
            self.train_env = TradingEnvBuilder.create_env(self.train_data, self.config)
            
            print("Creating testing environment...")
            self.test_env = TradingEnvBuilder.create_env(self.test_data, self.config)
            
            print("Environments created successfully")
            return self.train_env, self.test_env
        except Exception as e:
            print(f"Error setting up environments: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def train_model(self, total_timesteps=None):
        """
        Train the PPO model
        
        Args:
            total_timesteps (int, optional): Number of timesteps to train for
        """
        print("\n=== Step 3: Training PPO Model ===")
        
        # Use config value if not specified
        if total_timesteps is None:
            total_timesteps = self.config.TOTAL_TIMESTEPS
        
        # Create trainer
        self.trainer = PPOAgentTrainer(self.train_env, self.config.PPO_PARAMS)
        
        # Train model
        self.model = self.trainer.train(total_timesteps)
        
        return self.model
    
    def backtest_model(self):
        """
        Backtest the trained model
        """
        print("\n=== Step 4: Backtesting Model ===")
        
        # Run prediction
        self.account_value, self.actions = PPOAgentTrainer.predict(self.model, self.test_env)
        
        # Evaluate strategy
        self.account_value, agent_metrics = BacktestEvaluator.evaluate_strategy(self.account_value)
        
        return self.account_value, self.actions
    
    def fetch_benchmarks(self):
        """
        Fetch benchmark data for comparison
        """
        print("\n=== Step 5: Fetching Benchmark Data ===")
        
        # Fetch benchmark data
        raw_benchmarks = self.data_fetcher.fetch_benchmark_data(
            self.config.BENCHMARKS,
            self.config.TEST_START_DATE,
            self.config.TEST_END_DATE
        )
        
        # Process benchmarks
        self.benchmarks = BenchmarkCalculator.prepare_benchmarks(raw_benchmarks)
        
        return self.benchmarks
    
    def evaluate_performance(self):
        """
        Evaluate performance against benchmarks
        """
        print("\n=== Step 6: Evaluating Performance ===")
        
        # Get agent metrics
        _, agent_metrics = BacktestEvaluator.evaluate_strategy(self.account_value)
        
        # Get benchmark metrics
        benchmark_metrics = BacktestEvaluator.evaluate_benchmarks(self.benchmarks)
        
        # Compare performance
        self.performance_metrics = BacktestEvaluator.compare_performance(
            agent_metrics, benchmark_metrics
        )
        
        # Calculate correlation
        correlation = BenchmarkCalculator.calculate_correlation(
            self.account_value['daily_return'],
            self.benchmarks
        )
        
        # Calculate beta and alpha (using S&P 500 as market)
        if 'S&P 500' in self.benchmarks:
            market_returns = self.benchmarks['S&P 500']['daily_return']
            beta = BenchmarkCalculator.calculate_beta(self.account_value['daily_return'], market_returns)
            alpha = BenchmarkCalculator.calculate_alpha(self.account_value['daily_return'], market_returns)
            
            print(f"Beta (relative to S&P 500): {beta:.4f}")
            print(f"Alpha (annualized): {alpha:.4f}")
        
        print("\nPerformance Comparison:")
        print(self.performance_metrics)
        
        return self.performance_metrics, correlation
    
    def visualize_results(self):
        """
        Visualize backtest results
        """
        print("\n=== Step 7: Visualizing Results ===")
        
        # Create plotter
        plotter = PerformancePlotter(self.config)
        
        # Plot cumulative returns
        plotter.plot_cumulative_returns(
            self.account_value, 
            self.benchmarks,
            save_path="results/cumulative_returns.png"
        )
        
        # Plot rolling metrics
        plotter.plot_rolling_metrics(
            self.account_value,
            self.benchmarks,
            save_path="results/rolling_metrics.png"
        )
        
        # Plot correlation heatmap
        correlation = BenchmarkCalculator.calculate_correlation(
            self.account_value['daily_return'],
            self.benchmarks
        )
        plotter.plot_correlation_heatmap(
            correlation,
            save_path="results/correlation_heatmap.png"
        )
        
        # Plot performance metrics
        plotter.plot_performance_metrics(
            self.performance_metrics,
            save_path="results/performance_metrics.png"
        )
        
        print("Visualizations saved in 'results' directory")
    
    def save_results(self):
        """
        Save results to files
        """
        print("\n=== Step 8: Saving Results ===")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save performance metrics
        self.performance_metrics.to_csv(f"results/performance_metrics_{timestamp}.csv")
        
        # Save account value history
        self.account_value.to_csv(f"results/account_value_{timestamp}.csv")
        
        # Save trading actions
        if self.actions is not None:
            self.actions.to_csv(f"results/actions_{timestamp}.csv")
        
        print("Results saved in 'results' directory")
    
    def run_pipeline(self, total_timesteps=None):
        """
        Run the complete trading pipeline
        
        Args:
            total_timesteps (int, optional): Number of timesteps to train for
        """
        # Step 1: Fetch and process data
        self.fetch_and_process_data()
        
        # Step 2: Setup environments
        self.setup_environments()
        
        # Step 3: Train model
        self.train_model(total_timesteps)
        
        # Step 4: Backtest model
        self.backtest_model()
        
        # Step 5: Fetch benchmarks
        self.fetch_benchmarks()
        
        # Step 6: Evaluate performance
        self.evaluate_performance()
        
        # Step 7: Visualize results
        self.visualize_results()
        
        # Step 8: Save results
        self.save_results()
        
        print("\n=== Pipeline Complete ===")
        print(f"Check the 'results' directory for output files and visualizations")
        
        return self.performance_metrics

def main():
    """
    Main function to run the SPX trader
    """
    print("SPX Trading with PPO Agent")
    print("================================")
    
    # Get API keys from environment variables or use placeholders
    api_key = os.environ.get("APCA_API_KEY")
    secret_key = os.environ.get("APCA_API_SECRET")
    
    # Create trader
    trader = SPXTrader(api_key, secret_key)
    
    # Run pipeline
    trader.run_pipeline()

if __name__ == "__main__":
    main()