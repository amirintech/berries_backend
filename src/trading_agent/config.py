import torch

class Config:
    # Data settings
    TICKER = "SPX"
    INDICATORS = ["macd", "rsi", "cci", "dx"]  # Technical indicators to be used
    TRAIN_START_DATE = "2010-01-01"
    TRAIN_END_DATE = "2021-12-31"
    TEST_START_DATE = "2022-01-01" 
    TEST_END_DATE = "2023-12-31"

    # Environment settings
    INITIAL_AMOUNT = 10_000
    MAX_SHARES_PER_TRADE = 500
    BUY_COST_PCT = 1e-3
    SELL_COST_PCT = 1e-3
    REWARD_SCALING = 1e-4
    TURBULENCE_THRESHOLD = 0
    
    # Training settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOTAL_TIMESTEPS = 200000 if torch.cuda.is_available() else 100000
    
    # PPO parameters
    PPO_PARAMS = {
        "n_steps": 2048,
        "ent_coef": 0.05,
        "learning_rate": 0.001,
        "batch_size": 128,
        "device": DEVICE
    }
    
    # Benchmark tickers (Yahoo Finance format)
    BENCHMARKS = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT'
    }
    
    # Plot settings
    ROLLING_WINDOW = 30  # 30-day rolling window for metrics
    FIGURE_SIZE_MAIN = (15, 10)
    FIGURE_SIZE_MULTI = (15, 18)
    FIGURE_SIZE_HEATMAP = (10, 8)