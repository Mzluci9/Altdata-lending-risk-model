import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
def setup_logging(log_dir="logs"):
    """Set up logging to track info and errors in separate files."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "credit_risk_info.log")),
            logging.FileHandler(os.path.join(log_dir, "credit_risk_error.log")),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

logger = setup_logging()

def calculate_woe(data: pd.DataFrame, feature_col: str, target_col: str, epsilon=1e-6) -> pd.DataFrame:
    """
    Calculate Weight of Evidence (WoE) for a feature's bins relative to a binary target.

    Parameters:
    - data: DataFrame containing the feature and target columns.
    - feature_col: Name of the binned feature column.
    - target_col: Name of the binary target column (0 or 1).
    - epsilon: Small constant to avoid division by zero.

    Returns:
    - DataFrame with WoE values for each bin.
    """
    logger.info(f"Calculating WoE for feature: {feature_col}")
    
    try:
        # Validate inputs
        if feature_col not in data.columns or target_col not in data.columns:
            raise ValueError(f"Feature {feature_col} or target {target_col} not in DataFrame")
        if not data[target_col].isin([0, 1]).all():
            raise ValueError(f"Target column {target_col} must be binary (0 or 1)")
        
        # Calculate total good and bad counts
        total_good = data[target_col].sum()
        total_bad = len(data) - total_good
        
        if total_good == 0 or total_bad == 0:
            raise ValueError("Target column has no variation (all 0s or all 1s)")
        
        # Group by feature and calculate counts
        woe_df = data.groupby(feature_col)[target_col].agg(['count', 'sum'])
        woe_df = woe_df.rename(columns={'sum': 'good'})
        woe_df['bad'] = woe_df['count'] - woe_df['good']
        
        # Calculate distributions with epsilon to avoid zero division
        woe_df['good_dist'] = (woe_df['good'] + epsilon) / (total_good + epsilon)
        woe_df['bad_dist'] = (woe_df['bad'] + epsilon) / (total_bad + epsilon)
        woe_df['woe'] = np.log(woe_df['good_dist'] / woe_df['bad_dist'])
        
        logger.info(f"WoE calculation successful for {feature_col}")
        return woe_df[['woe']]
    
    except Exception as e:
        logger.error(f"Error in WoE calculation for {feature_col}: {str(e)}")
        raise

def visualize_woe(woe_df: pd.DataFrame, feature_name: str, save_path: str = None):
    """
    Visualize WoE values for a feature using a bar plot.

    Parameters:
    - woe_df: DataFrame with WoE values (index: bins, column: 'woe').
    - feature_name: Name of the feature for plot title.
    - save_path: Optional path to save the plot.
    """
    logger.info(f"Creating WoE plot for {feature_name}")
    
    try:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=woe_df.index, y=woe_df['woe'], hue=woe_df.index, palette='viridis', legend=False)
        plt.title(f'Weight of Evidence for {feature_name}', fontsize=14)
        plt.xlabel(f'{feature_name} Bins', fontsize=12)
        plt.ylabel('WoE', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"WoE plot saved to {save_path}")
        plt.show()
        
        logger.info(f"WoE plot completed for {feature_name}")
    
    except Exception as e:
        logger.error(f"Error plotting WoE for {feature_name}: {str(e)}")
        raise