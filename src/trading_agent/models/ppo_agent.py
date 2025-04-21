from finrl.agents.stablebaselines3.models import DRLAgent

class PPOAgentTrainer:
    """
    Class for training PPO agent
    """
    def __init__(self, env, model_params):
        """
        Initialize with environment and model parameters
        
        Args:
            env: Training environment
            model_params (dict): PPO hyperparameters
        """
        self.env = env
        self.model_params = model_params
        self.agent = DRLAgent(env=self.env)
        self.model = None
    
    def train(self, total_timesteps=100000, tb_log_name='ppo'):
        """
        Train the PPO model
        
        Args:
            total_timesteps (int): Number of timesteps to train for
            tb_log_name (str): TensorBoard log name
            
        Returns:
            Trained model
        """
        print(f"Training PPO model for {total_timesteps} timesteps...")
        
        # Get model
        model = self.agent.get_model("ppo", model_kwargs=self.model_params)
        
        # Train the model
        trained_model = self.agent.train_model(
            model=model,
            tb_log_name=tb_log_name,
            total_timesteps=total_timesteps
        )
        
        self.model = trained_model
        
        return trained_model

    @staticmethod
    def predict(model, env):
        """
        Run prediction using the trained model
        
        Args:
            model: Trained PPO model
            env: Testing environment
            
        Returns:
            tuple: (account value DataFrame, actions DataFrame)
        """
        print("Running prediction with trained model...")
        
        df_account_value, df_actions = DRLAgent.DRL_prediction(
            model=model,
            environment=env
        )
        
        return df_account_value, df_actions
