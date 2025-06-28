# Altdata-lending-risk-model(
Credit Scoring Business Understanding)
A machine learning model that predicts credit risk using alternative data sources for improved borrower assessment.



this section address three key questions:
1.	Basel II Accord’s Influence: The Basel II Accord emphasizes risk measurement and transparency, requiring an interpretable and well-documented model to ensure regulatory compliance and stakeholder trust. My model will prioritize explainability (e.g., using Logistic Regression with Weight of Evidence) to align with these requirements.
2.	Need for a Proxy Variable: Since the dataset lacks a direct “default” label, I plan to create a proxy variable based on customer behavior (e.g., RFM metrics or fraud indicators). Potential risks include misfixed if the proxy does not accurately reflect true default behavior, which could lead to incorrect loan decisions.
3.	Model Trade-offs: Simple models like Logistic Regression are interpretable and compliant but may lack predictive power. Complex models like Gradient Boosting offer higher performance but are harder to explain in a regulated context, requiring additional interpretability tools (e.g., SHAP).


## Feature Engineering

The feature engineering pipeline is implemented in `src/data_processing.py` using a `FeatureEngineer` class. The class performs the following transformations:
- **Aggregate Features**: Creates RFM-like features (e.g., total_amount, transaction_count) per CustomerId.
- **Time-Based Features**: Extracts hour, day, month, year, and day of week from TransactionStartTime.
- **Categorical Encoding**: Uses one-hot encoding for ProductCategory and ChannelId, and label encoding for ProviderId.
- **Missing Value Imputation**: Imputes numerical features with median and categorical with mode.
- **Standardization**: Scales numerical features to mean 0 and standard deviation 1.

Processed data is saved to `data/processed/processed_data.csv`.
