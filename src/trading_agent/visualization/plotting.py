import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PerformancePlotter:
    """
    Class for plotting performance metrics
    """
    def __init__(self, config):
        """
        Initialize with configuration settings
        """
        self.config = config
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = config.FIGURE_SIZE_MAIN
    
    def plot_cumulative_returns(self, df_account_value, benchmarks, save_path=None):
        """
        Plot cumulative returns comparison
        
        Args:
            df_account_value (DataFrame): Agent's account value
            benchmarks (dict): Dictionary of benchmark DataFrames
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        plt.figure(figsize=self.config.FIGURE_SIZE_MAIN)
        
        # Plot PPO Agent performance
        plt.plot(df_account_value.index, df_account_value['cumulative_return'], 
                 label='PPO Agent', linewidth=2)
        
        # Plot benchmarks
        for name, benchmark in benchmarks.items():
            plt.plot(benchmark.index, benchmark['cumulative_return'], 
                     label=name, linewidth=1.5, alpha=0.8)
        
        plt.title('PPO Agent vs Benchmark Strategies (Cumulative Returns)', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()
    
    def plot_rolling_metrics(self, df_account_value, benchmarks, save_path=None):
        """
        Plot rolling performance metrics
        
        Args:
            df_account_value (DataFrame): Agent's account value
            benchmarks (dict): Dictionary of benchmark DataFrames
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, axes = plt.subplots(3, 1, figsize=self.config.FIGURE_SIZE_MULTI)
        
        # Get window size from config
        window = self.config.ROLLING_WINDOW
        
        # 1. Rolling Sharpe Ratio
        ax = axes[0]
        # Calculate rolling Sharpe for PPO Agent
        rolling_returns = df_account_value['daily_return'].fillna(0)
        rolling_sharpe = (rolling_returns.rolling(window=window).mean() / 
                         rolling_returns.rolling(window=window).std()) * np.sqrt(252)
        ax.plot(rolling_sharpe.index, rolling_sharpe, label='PPO Agent', linewidth=2)
        
        # Calculate rolling Sharpe for benchmarks
        for name, benchmark in benchmarks.items():
            benchmark_returns = benchmark['daily_return'].fillna(0)
            benchmark_sharpe = (benchmark_returns.rolling(window=window).mean() / 
                               benchmark_returns.rolling(window=window).std()) * np.sqrt(252)
            ax.plot(benchmark_sharpe.index, benchmark_sharpe, label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_title(f'{window}-Day Rolling Sharpe Ratio', fontsize=15)
        ax.set_ylabel('Sharpe Ratio', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        ax = axes[1]
        # Calculate rolling volatility for PPO Agent
        rolling_vol = rolling_returns.rolling(window=window).std() * np.sqrt(252)
        ax.plot(rolling_vol.index, rolling_vol, label='PPO Agent', linewidth=2)
        
        # Calculate rolling volatility for benchmarks
        for name, benchmark in benchmarks.items():
            benchmark_returns = benchmark['daily_return'].fillna(0)
            benchmark_vol = benchmark_returns.rolling(window=window).std() * np.sqrt(252)
            ax.plot(benchmark_vol.index, benchmark_vol, label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_title(f'{window}-Day Rolling Volatility', fontsize=15)
        ax.set_ylabel('Annualized Volatility', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax = axes[2]
        # Calculate drawdown for PPO Agent
        cumulative = (1 + rolling_returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = ((cumulative / peak) - 1)
        ax.plot(drawdown.index, drawdown, label='PPO Agent', linewidth=2)
        
        # Calculate drawdown for benchmarks
        for name, benchmark in benchmarks.items():
            benchmark_returns = benchmark['daily_return'].fillna(0)
            benchmark_cumulative = (1 + benchmark_returns).cumprod()
            benchmark_peak = benchmark_cumulative.expanding(min_periods=1).max()
            benchmark_drawdown = ((benchmark_cumulative / benchmark_peak) - 1)
            ax.plot(benchmark_drawdown.index, benchmark_drawdown, label=name, linewidth=1.5, alpha=0.8)
        
        ax.set_title('Drawdown', fontsize=15)
        ax.set_ylabel('Drawdown', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    def plot_correlation_heatmap(self, correlation_matrix, save_path=None):
        """
        Plot correlation heatmap
        
        Args:
            correlation_matrix (DataFrame): Correlation matrix
            save_path (str, optional): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        plt.figure(figsize=self.config.FIGURE_SIZE_HEATMAP)
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                   vmin=-1, vmax=1, linewidths=.5)
        
        plt.title('Correlation of Returns', fontsize=15)
        plt.tight_layout()
        
        # Save the figure if path is provided
        if save_path:
            plt.savefig(save_path)
        
        return plt.gcf()
    
    def plot_performance_metrics(self, performance_df, save_path=None):
        """
        Plot performance metrics comparison
        
        Args:
            performance_df (DataFrame): Performance metrics comparison
            save_path (str, optional): Path to save the figure
            
        Returns:
            list: List of figure objects
        """
        figures = []
        
        # Create a separate plot for each metric
        metrics = performance_df.columns
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            # Sort values for better visualization
            sorted_performance = performance_df[metric].sort_values(ascending=False)
            
            # Create bar chart
            ax = sorted_performance.plot(kind='bar', color=sns.color_palette("viridis", len(sorted_performance)))
            
            plt.title(f'Comparison of {metric}', fontsize=15)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Add values on top of bars
            for i, v in enumerate(sorted_performance):
                ax.text(i, v + (v * 0.01 if v > 0 else v * 0.01), 
                       f'{v:.4f}', ha='center', fontsize=10)
            
            # Save the figure if path is provided
            if save_path:
                metric_name = metric.lower().replace(' ', '_')
                plt.savefig(f"{save_path.split('.')[0]}_{metric_name}.png")
            
            figures.append(plt.gcf())
        
        return figures