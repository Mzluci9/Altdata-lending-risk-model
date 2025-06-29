import pandas as pd
import numpy as np
import logging
import os
from typing import Optional

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'rfms_info.log')),
        logging.FileHandler(os.path.join(log_dir, 'rfms_error.log'))
    ]
)
logger = logging.getLogger(__name__)

class RFMSProcessor:
    """Class to compute RFMS (Recency, Frequency, Monetary, Stability) scores."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize RFMSProcessor with transaction data.

        Args:
            data (pd.DataFrame): Input DataFrame with transaction details.
        """
        self.data = data.copy()
        logger.info(f"RFMSProcessor initialized with data shape: {self.data.shape}")

    def compute_recency(self) -> pd.DataFrame:
        """
        Calculate recency as days since the last transaction.

        Returns:
            pd.DataFrame: DataFrame with 'Recency' column.
        """
        logger.info("Computing recency...")
        self.data['TransactionStartTime'] = pd.to_datetime(self.data['TransactionStartTime'])
        max_date = self.data['TransactionStartTime'].max()
        self.data['Recency'] = (max_date - self.data['TransactionStartTime']).dt.days
        logger.info("Recency computed successfully")
        return self.data

    def compute_frequency_monetary(self) -> pd.DataFrame:
        """
        Calculate frequency (transaction count) and monetary (total amount) per customer.

        Returns:
            pd.DataFrame: DataFrame with 'Frequency' and 'Monetary' columns.
        """
        logger.info("Computing frequency and monetary values...")
        agg_df = self.data.groupby('CustomerId').agg(
            Frequency=('TransactionId', 'count'),
            Monetary=('Amount', 'sum')
        ).reset_index()
        self.data = self.data.merge(agg_df, on='CustomerId', how='left')
        logger.info("Frequency and monetary values computed successfully")
        return self.data

    def assign_rfms_label(self, recency_threshold: Optional[float] = None,
                         frequency_threshold: Optional[float] = None,
                         monetary_threshold: Optional[float] = None) -> pd.DataFrame:
        """
        Assign RFMS labels based on thresholds (default: median).

        Args:
            recency_threshold (float, optional): Threshold for recency.
            frequency_threshold (float, optional): Threshold for frequency.
            monetary_threshold (float, optional): Threshold for monetary value.

        Returns:
            pd.DataFrame: DataFrame with 'RFMS_Label' column (0 = Good, 1 = Bad).
        """
        logger.info("Assigning RFMS labels...")
        recency_threshold = recency_threshold or self.data['Recency'].median()
        frequency_threshold = frequency_threshold or self.data['Frequency'].median()
        monetary_threshold = monetary_threshold or self.data['Monetary'].median()

        self.data['RFMS_Label'] = np.where(
            (self.data['Recency'] <= recency_threshold) &
            (self.data['Frequency'] >= frequency_threshold) &
            (self.data['Monetary'] >= monetary_threshold),
            0,  # Good
            1   # Bad
        )
        logger.info("RFMS labels assigned successfully")
        return self.data

    def assign_user_labels(self) -> pd.DataFrame:
        """
        Convert RFMS labels to user-friendly 'Good' or 'Bad' labels.

        Returns:
            pd.DataFrame: DataFrame with 'User_Label' column.
        """
        logger.info("Assigning user labels...")
        self.data['User_Label'] = self.data['RFMS_Label'].map({0: 'Good', 1: 'Bad'})
        logger.info("User labels assigned successfully")
        return self.data

    def calculate_rfms(self) -> pd.DataFrame:
        """
        Compute all RFMS metrics and labels.

        Returns:
            pd.DataFrame: DataFrame with RFMS features and labels.
        """
        logger.info("Starting RFMS calculation...")
        self.compute_recency()
        self.compute_frequency_monetary()
        self.assign_rfms_label()
        self.assign_user_labels()
        logger.info("RFMS calculation completed")
        return self.data