import pandas as pd
import numpy as np

class BacktestEvaluator:
    """
    Class for backtesting and performance evaluation
    """
    @staticmethod
    def calculate_metrics(df_returns):
        """
        Calculate performance metrics from daily returns
        
        Args:
            df_returns (Series): Daily returns series
            
        Returns:
            dict: Performance metrics
        """
        # Fill NaN values
        returns = df_returns.fillna(0)
        
        # Calculate metrics
        total_return = ((1 + returns).cumprod() - 1).iloc[-1]
        annual_return = ((1 + total_return) ** (252 / len(returns)) - 1)
        
        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming 0 risk-free rate)
        sharpe = annual_return / volatility if volatility != 0 else 0
        
        # Calculate max drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio (downside deviation)
        downside_returns = returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino = annual_return / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar
        }
    
    @staticmethod
    def evaluate_strategy(df_account_value):
        """
        Evaluate the agent's strategy
        
        Args:
            df_account_value (DataFrame): Account value over time
            
        Returns:
            tuple: (daily returns, metrics)
        """
        print("Evaluating agent's strategy...")
        
        # Calculate daily returns
        df_account_value['daily_return'] = df_account_value['account_value'].pct_change(1)
        
        # Calculate cumulative returns
        df_account_value['cumulative_return'] = (1 + df_account_value['daily_return'].fillna(0)).cumprod() - 1
        
        # Calculate performance metrics
        metrics = BacktestEvaluator.calculate_metrics(df_account_value['daily_return'])
        
        return df_account_value, metrics
    
    @staticmethod
    def evaluate_benchmarks(benchmarks):
        """
        Evaluate benchmark strategies
        
        Args:
            benchmarks (dict): Dictionary of benchmark DataFrames
            
        Returns:
            dict: Benchmark metrics
        """
        print("Evaluating benchmark strategies...")
        
        benchmark_metrics = {}
        
        for name, benchmark in benchmarks.items():
            # Calculate cumulative returns if not already calculated
            if 'cumulative_return' not in benchmark.columns:
                benchmark['cumulative_return'] = (1 + benchmark['daily_return'].fillna(0)).cumprod() - 1
            
            # Calculate metrics
            metrics = BacktestEvaluator.calculate_metrics(benchmark['daily_return'])
            benchmark_metrics[name] = metrics
        
        return benchmark_metrics
    
    @staticmethod
    def compare_performance(agent_metrics, benchmark_metrics):
        """
        Compare agent performance with benchmarks
        
        Args:
            agent_metrics (dict): Agent performance metrics
            benchmark_metrics (dict): Benchmark performance metrics
            
        Returns:
            DataFrame: Performance comparison table
        """
        print("Comparing performance...")
        
        # Combine metrics
        all_metrics = {'PPO Agent': agent_metrics}
        all_metrics.update(benchmark_metrics)
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(all_metrics).T
        
        return performance_df