# spx_trader/evaluation/benchmarks.py

import pandas as pd
import numpy as np

class BenchmarkCalculator:
    """
    Class for creating and analyzing benchmark strategies
    """
    @staticmethod
    def prepare_benchmarks(benchmarks):
        """
        Prepare benchmark data for comparison
        
        Args:
            benchmarks (dict): Dictionary of benchmark DataFrames
            
        Returns:
            dict: Processed benchmark DataFrames
        """
        print("Preparing benchmark data...")
        
        processed_benchmarks = {}
        
        for name, benchmark in benchmarks.items():
            # Ensure daily returns are calculated
            if 'daily_return' not in benchmark.columns:
                benchmark['daily_return'] = benchmark['Close'].pct_change(1)
            
            # Calculate cumulative returns
            benchmark['cumulative_return'] = (1 + benchmark['daily_return'].fillna(0)).cumprod() - 1
            
            processed_benchmarks[name] = benchmark
        
        return processed_benchmarks
    
    @staticmethod
    def calculate_correlation(agent_returns, benchmarks):
        """
        Calculate correlation between agent and benchmark returns
        
        Args:
            agent_returns (Series): Agent daily returns
            benchmarks (dict): Dictionary of benchmark DataFrames
            
        Returns:
            DataFrame: Correlation matrix
        """
        print("Calculating return correlations...")
        
        # Create correlation dataframe
        corr_data = pd.DataFrame({'PPO Agent': agent_returns.fillna(0)})
        
        for name, benchmark in benchmarks.items():
            # Align dates
            aligned_returns = benchmark['daily_return'].reindex(corr_data.index).fillna(0)
            corr_data[name] = aligned_returns
        
        # Calculate correlation matrix
        correlation = corr_data.corr()
        
        return correlation
    
    @staticmethod
    def calculate_beta(agent_returns, market_returns):
        """
        Calculate beta of agent returns relative to market
        
        Args:
            agent_returns (Series): Agent daily returns
            market_returns (Series): Market daily returns
            
        Returns:
            float: Beta value
        """
        # Align dates
        aligned_agent = agent_returns.fillna(0)
        aligned_market = market_returns.reindex(aligned_agent.index).fillna(0)
        
        # Calculate covariance and variance
        covariance = aligned_agent.cov(aligned_market)
        variance = aligned_market.var()
        
        # Calculate beta
        beta = covariance / variance if variance != 0 else 0
        
        return beta
    
    @staticmethod
    def calculate_alpha(agent_returns, market_returns, risk_free_rate=0):
        """
        Calculate Jensen's alpha
        
        Args:
            agent_returns (Series): Agent daily returns
            market_returns (Series): Market daily returns
            risk_free_rate (float): Risk-free rate (annual)
            
        Returns:
            float: Alpha value
        """
        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate beta
        beta = BenchmarkCalculator.calculate_beta(agent_returns, market_returns)
        
        # Calculate average returns
        avg_agent = agent_returns.fillna(0).mean()
        avg_market = market_returns.fillna(0).mean()
        
        # Calculate alpha (daily basis)
        alpha = avg_agent - (daily_rf + beta * (avg_market - daily_rf))
        
        # Annualize alpha
        annual_alpha = ((1 + alpha) ** 252) - 1
        
        return annual_alpha