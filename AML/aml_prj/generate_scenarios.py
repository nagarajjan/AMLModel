import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from datetime import datetime, timedelta

def create_demo_data_files():
    """
    Generates and saves the demo datasets into CSV files.
    - An initial dataset is created and split into training and testing sets.
    - A new dataset is created and split into new training and holdout sets.
    """
    print("Generating demo data files...")
    if not os.path.exists('data'):
        os.makedirs('data')

    # --- Initial Data ---
    size = 1000
    np.random.seed(42)
    initial_transactions = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.Series(range(size)), unit='h', origin='2024-01-01'),
        'transaction_amount': np.random.rand(size) * 100,
        'customer_id': np.random.randint(1, size/10, size),
        'other_feature': np.random.randn(size),
        'transaction_type': np.random.choice(['debit', 'credit', 'transfer'], size),
        'merchant_category': np.random.choice(['retail', 'services', 'utility'], size)
    })
    # Inject some initial anomalies
    initial_transactions.iloc[np.random.randint(0, size, 5), 1] = np.random.rand(5) * 1000

    # Split initial data into train and test sets
    train_data, test_data = train_test_split(initial_transactions, test_size=0.2, random_state=42)
    train_data.to_csv('data/initial_train.csv', index=False)
    test_data.to_csv('data/initial_test.csv', index=False)

    # --- New Data for Retraining ---
    new_data_size = 200
    np.random.seed(43)
    new_transactions = pd.DataFrame({
        'timestamp': pd.to_datetime(pd.Series(range(size, size + new_data_size)), unit='h', origin='2024-01-01'),
        'transaction_amount': np.random.rand(new_data_size) * 120,
        'customer_id': np.random.randint(1, size/10, new_data_size),
        'other_feature': np.random.randn(new_data_size) + 1,
        'transaction_type': np.random.choice(['debit', 'credit', 'transfer'], new_data_size),
        'merchant_category': np.random.choice(['retail', 'services', 'utility'], new_data_size)
    })
    # Inject new, more pronounced anomalies
    new_transactions.iloc[np.random.randint(0, new_data_size, 2), 1] = np.random.rand(2) * 2000

    new_retrain_data, new_holdout_data = train_test_split(new_transactions, test_size=0.2, random_state=42)
    new_retrain_data.to_csv('data/new_data.csv', index=False)
    new_holdout_data.to_csv('data/new_data_holdout.csv', index=False)

    print("Demo data files created in the 'data' directory.")

def create_scenario_file():
    """
    Generates a CSV file containing transactions designed to trigger specific AML/Fraud scenarios.
    """
    print("Creating AML scenario file...")
    scenarios = []
    base_timestamp = datetime(2025, 9, 29, 10, 0, 0)
    customer_id_large = 999
    customer_id_frequent = 1000
    customer_id_unusual_time = 1002

    # Scenario 1: Large Transaction
    scenarios.append({
        'timestamp': base_timestamp,
        'transaction_amount': 50000.00,
        'customer_id': customer_id_large,
        'other_feature': 1.0,
        'transaction_type': 'debit',
        'merchant_category': 'investment',
        'scenario_type': 'Large Transaction'
    })

    # Scenario 2: Frequent Small Transactions
    for i in range(10):
        scenarios.append({
            'timestamp': base_timestamp + timedelta(minutes=i*2),
            'transaction_amount': 50.00,
            'customer_id': customer_id_frequent,
            'other_feature': -0.2,
            'transaction_type': 'purchase',
            'merchant_category': 'online retail',
            'scenario_type': 'Frequent Small Transactions'
        })

    # Scenario 3: Unusual Transaction Time
    unusual_time_timestamp = datetime(2025, 9, 30, 3, 0, 0) # 3 AM
    scenarios.append({
        'timestamp': unusual_time_timestamp,
        'transaction_amount': 1500.00,
        'customer_id': customer_id_unusual_time,
        'other_feature': 0.0,
        'transaction_type': 'debit',
        'merchant_category': 'online services',
        'scenario_type': 'Unusual Transaction Time'
    })

    scenario_df = pd.DataFrame(scenarios)
    scenario_df.to_csv('data/aml_scenario.csv', index=False)
    print("AML scenario file created: data/aml_scenario.csv")

if __name__ == '__main__':
    create_demo_data_files()
    create_scenario_file()

