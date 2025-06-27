import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data_processing import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, -50, 200],
        'TransactionId': ['T1', 'T2', 'T3'],
        'ProductCategory': ['cat1', 'cat1', 'cat2'],
        'ChannelId': ['web', 'mobile', 'web'],
        'ProviderId': ['P1', 'P1', 'P2'],
        'TransactionStartTime': ['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00']
    })

def test_create_aggregate_features(sample_data):
    engineer = FeatureEngineer(input_path=None, output_path=None)
    engineer.df = sample_data
    engineer.create_aggregate_features()
    assert 'total_amount' in engineer.df.columns
    assert engineer.df[engineer.df['CustomerId'] == 1]['total_amount'].iloc[0] == 50

def test_extract_time_features(sample_data):
    engineer = FeatureEngineer(input_path=None, output_path=None)
    engineer.df = sample_data
    engineer.extract_time_features()
    assert 'transaction_hour' in engineer.df.columns
    assert engineer.df['transaction_hour'].iloc[0] == 10