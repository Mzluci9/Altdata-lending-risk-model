import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
from datetime import datetime
from src.F_data_processing import (
    AggregateFeatures,
    ExtractingFeatures,
    normalize_numerical_features,
    normalize_columns
)
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, -50, 200],
        'TransactionId': ['T1', 'T2', 'T3'],
        'TransactionStartTime': [
            '2023-01-01 10:00:00',
            '2023-01-02 12:00:00',
            '2023-01-03 14:00:00'
        ]
    })


def test_sum_all_transactions(sample_df):
    af = AggregateFeatures(sample_df.copy())
    af.sum_all_transactions()
    df = af.get_dataframe()
    assert 'TotalTransactionAmount' in df.columns
    assert df[df['CustomerId'] == 1]['TotalTransactionAmount'].iloc[0] == 50


def test_average_transaction_amount(sample_df):
    af = AggregateFeatures(sample_df.copy())
    af.average_transaction_amount()
    df = af.get_dataframe()
    assert 'AverageTransactionAmount' in df.columns
    assert df[df['CustomerId'] == 1]['AverageTransactionAmount'].iloc[0] == 25


def test_transaction_count(sample_df):
    af = AggregateFeatures(sample_df.copy())
    af.transaction_count()
    df = af.get_dataframe()
    assert 'TotalTransactions' in df.columns
    assert df[df['CustomerId'] == 1]['TotalTransactions'].iloc[0] == 2


def test_standard_deviation_amount(sample_df):
    af = AggregateFeatures(sample_df.copy())
    af.standard_deviation_amount()
    df = af.get_dataframe()
    assert 'StdTransactionAmount' in df.columns
    std_val = df[df['CustomerId'] == 1]['StdTransactionAmount'].iloc[0]
    assert round(std_val, 2) == 106.07  # std of [100, -50]


def test_transaction_hour(sample_df):
    ef = ExtractingFeatures(sample_df.copy())
    ef.transaction_hour()
    df = ef.get_dataframe()
    assert 'TransactionHour' in df.columns
    assert df['TransactionHour'].iloc[0] == 10


def test_transaction_day(sample_df):
    ef = ExtractingFeatures(sample_df.copy())
    ef.transaction_hour()
    ef.transaction_day()
    df = ef.get_dataframe()
    assert 'TransactionDay' in df.columns
    assert df['TransactionDay'].iloc[0] == 1


def test_transaction_month(sample_df):
    ef = ExtractingFeatures(sample_df.copy())
    ef.transaction_hour()
    ef.transaction_month()
    df = ef.get_dataframe()
    assert 'TransactionMonth' in df.columns
    assert df['TransactionMonth'].iloc[0] == 1


def test_transaction_year(sample_df):
    ef = ExtractingFeatures(sample_df.copy())
    ef.transaction_hour()
    ef.transaction_year()
    df = ef.get_dataframe()
    assert 'TransactionYear' in df.columns
    assert df['TransactionYear'].iloc[0] == 2023


def test_normalize_numerical_features(sample_df):
    df = sample_df.copy()
    df['TransactionId'] = df['TransactionId'].astype(str)
    df = normalize_numerical_features(df, numerical_cols=['Amount'], method='standardize')
    assert abs(df['Amount'].mean()) < 1e-6  # standardized mean ≈ 0


def test_normalize_columns(sample_df):
    df = sample_df.copy()
    df = normalize_columns(df, columns=['Amount'])
    assert df['Amount'].max() <= 1
    assert df['Amount'].min() >= 0


