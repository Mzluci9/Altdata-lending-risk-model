import pandas as pd
import numpy as np
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

class FeatureEngineer:
    """A class to handle feature engineering for the credit risk model."""

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self

    def create_aggregate_features(self):
        agg_df = self.df.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'TransactionId': 'nunique',
            'ProductCategory': 'nunique'
        }).reset_index()

        # Flatten column names
        agg_df.columns = ['CustomerId', 'total_amount', 'avg_amount', 'std_amount',
                          'amount_count', 'unique_transactions', 'unique_products']

        self.df = self.df.merge(agg_df, on='CustomerId', how='left')
        return self

    def extract_time_features(self):
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce')
        self.df['transaction_hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['transaction_day'] = self.df['TransactionStartTime'].dt.day
        self.df['transaction_month'] = self.df['TransactionStartTime'].dt.month
        self.df['transaction_year'] = self.df['TransactionStartTime'].dt.year
        self.df['transaction_dayofweek'] = self.df['TransactionStartTime'].dt.dayofweek
        return self

    def encode_categorical(self):
        cat_features = ['ProductCategory', 'ChannelId']
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe_df = pd.DataFrame(ohe.fit_transform(self.df[cat_features]),
                              columns=ohe.get_feature_names_out(cat_features))

        self.df = pd.concat([self.df.drop(cat_features, axis=1), ohe_df], axis=1)

        if 'ProviderId' in self.df.columns:
            le = LabelEncoder()
            self.df['ProviderId'] = le.fit_transform(self.df['ProviderId'].astype(str))
        return self

    def impute_missing(self):
        num_imputer = SimpleImputer(strategy='median')
        cat_imputer = SimpleImputer(strategy='most_frequent')

        numerical_cols = self.df.select_dtypes(include='number').columns
        categorical_cols = self.df.select_dtypes(include='object').columns

        self.df[numerical_cols] = num_imputer.fit_transform(self.df[numerical_cols])
        self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])
        return self

    def standardize_features(self):
        scaler = StandardScaler()
        numerical_cols = self.df.select_dtypes(include='number').columns
        self.df[numerical_cols] = scaler.fit_transform(self.df[numerical_cols])
        return self

    def save_processed_data(self):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.df.to_csv(self.output_path, index=False)
        return self

    def run_pipeline(self):
        return (
            self.load_data()
                .create_aggregate_features()
                .extract_time_features()
                .encode_categorical()
                .impute_missing()
                .standardize_features()
                .save_processed_data()
        )


# Example usage
if __name__ == "__main__":
    input_path = "data/data.csv"
    output_path = "data/processed/processed_data.csv"

    engineer = FeatureEngineer(input_path=input_path, output_path=output_path)
    engineer.run_pipeline()
