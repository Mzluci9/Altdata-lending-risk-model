# Altdata-lending-risk-model
A machine learning model that predicts credit risk using alternative data sources for improved borrower assessment.








## Feature Engineering

The feature engineering pipeline is implemented in `src/data_processing.py` using a `FeatureEngineer` class. The class performs the following transformations:
- **Aggregate Features**: Creates RFM-like features (e.g., total_amount, transaction_count) per CustomerId.
- **Time-Based Features**: Extracts hour, day, month, year, and day of week from TransactionStartTime.
- **Categorical Encoding**: Uses one-hot encoding for ProductCategory and ChannelId, and label encoding for ProviderId.
- **Missing Value Imputation**: Imputes numerical features with median and categorical with mode.
- **Standardization**: Scales numerical features to mean 0 and standard deviation 1.

Processed data is saved to `data/processed/processed_data.csv`.