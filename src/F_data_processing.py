import pandas as pd
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Set up logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

log_file_info = os.path.join(LOG_DIR, 'info.log')
log_file_error = os.path.join(LOG_DIR, 'error.log')

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)
info_handler.setFormatter(formatter)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)


class AggregateFeatures:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("AggregateFeatures initialized with dataframe of shape %s", df.shape)

    def sum_all_transactions(self):
        try:
            logger.info("Summing all transactions per customer.")
            total = self.df.groupby('CustomerId')['Amount'].sum().reset_index()
            total.rename(columns={'Amount': 'TotalTransactionAmount'}, inplace=True)
            self.df = self.df.merge(total, on='CustomerId', how='left')
        except Exception as e:
            logger.error("Error in sum_all_transactions: %s", e)

    def average_transaction_amount(self):
        try:
            logger.info("Calculating average transaction amount per customer.")
            avg = self.df.groupby('CustomerId')['Amount'].mean().reset_index()
            avg.rename(columns={'Amount': 'AverageTransactionAmount'}, inplace=True)
            self.df = self.df.merge(avg, on='CustomerId', how='left')
        except Exception as e:
            logger.error("Error in average_transaction_amount: %s", e)

    def transaction_count(self):
        try:
            logger.info("Counting transactions per customer.")
            count = self.df.groupby('CustomerId')['TransactionId'].count().reset_index()
            count.rename(columns={'TransactionId': 'TotalTransactions'}, inplace=True)
            self.df = self.df.merge(count, on='CustomerId', how='left')
        except Exception as e:
            logger.error("Error in transaction_count: %s", e)

    def standard_deviation_amount(self):
        try:
            logger.info("Calculating standard deviation of transaction amount.")
            std = self.df.groupby('CustomerId')['Amount'].std().reset_index()
            std.rename(columns={'Amount': 'StdTransactionAmount'}, inplace=True)
            self.df = self.df.merge(std, on='CustomerId', how='left')
        except Exception as e:
            logger.error("Error in standard_deviation_amount: %s", e)

    def get_dataframe(self):
        return self.df


class ExtractingFeatures:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        logger.info("ExtractingFeatures initialized with dataframe of shape %s", df.shape)

    def transaction_hour(self):
        try:
            logger.info("Extracting transaction hour.")
            self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'])
            self.df['TransactionHour'] = self.df['TransactionStartTime'].dt.hour
        except Exception as e:
            logger.error("Error in transaction_hour: %s", e)

    def transaction_day(self):
        try:
            logger.info("Extracting transaction day.")
            self.df['TransactionDay'] = self.df['TransactionStartTime'].dt.day
        except Exception as e:
            logger.error("Error in transaction_day: %s", e)

    def transaction_month(self):
        try:
            logger.info("Extracting transaction month.")
            self.df['TransactionMonth'] = self.df['TransactionStartTime'].dt.month
        except Exception as e:
            logger.error("Error in transaction_month: %s", e)

    def transaction_year(self):
        try:
            logger.info("Extracting transaction year.")
            self.df['TransactionYear'] = self.df['TransactionStartTime'].dt.year
        except Exception as e:
            logger.error("Error in transaction_year: %s", e)

    def get_dataframe(self):
        return self.df


def normalize_numerical_features(df, numerical_cols, method='standardize'):
    """
    Normalize or standardize numerical features.
    """
    try:
        logger.info("Scaling numerical features: %s using method: %s", numerical_cols, method)
        scaler = StandardScaler() if method == 'standardize' else MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        df.set_index('TransactionId', inplace=True)
        logger.info("Numerical features normalized successfully.")
        return df
    except Exception as e:
        logger.error("Error in normalize_numerical_features: %s", e)
        raise


def normalize_columns(df, columns):
    """
    Normalize specific columns using MinMaxScaler.
    """
    try:
        logger.info("Normalizing columns: %s", columns)
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        logger.info("Columns normalized successfully.")
        return df
    except Exception as e:
        logger.error("Error in normalize_columns: %s", e)
        raise
