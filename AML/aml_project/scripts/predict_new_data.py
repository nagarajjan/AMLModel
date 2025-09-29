import pandas as pd
import numpy as np
from scripts.model_engine import AMLEngine, FraudEngine

def simulate_new_data(num_records=500, num_anomalies_aml=10, num_anomalies_fraud=5, seed=43):
    """Generates synthetic data to represent new transactions."""
    np.random.seed(seed)
    # Generate new AML data
    aml_data = {
        'Timestamp': pd.to_datetime('2025-01-01') + pd.to_timedelta(np.random.randint(0, 365, size=num_records), unit='D'),
        'From': np.random.randint(1000, 5000, size=num_records),
        'To': np.random.randint(1000, 5000, size=num_records),
        'FromBank': np.random.choice(['BankA', 'BankB'], size=num_records),
        'ToBank': np.random.choice(['BankA', 'BankB'], size=num_records),
        'type': np.random.choice(['transfer', 'withdrawal', 'deposit', 'payment'], size=num_records),
        'Amount': np.random.lognormal(mean=7, sigma=1.5, size=num_records),
        'Is Laundering': np.random.choice([0, 1], size=num_records, p=[0.98, 0.02])
    }
    aml_df = pd.DataFrame(aml_data)
    
    # Generate new Fraud data (mimicking creditcard.csv)
    fraud_data = {
        'Time': np.random.randint(0, 172792, size=num_records),
        'V1': np.random.randn(num_records), 'V2': np.random.randn(num_records), 'V3': np.random.randn(num_records),
        'Amount': np.random.lognormal(mean=2, sigma=1, size=num_records),
        'Class': np.random.choice([0, 1], size=num_records, p=[0.995, 0.005])
    }
    fraud_df = pd.DataFrame(fraud_data)
    
    return aml_df, fraud_df

def predict_on_new_data():
    """Predicts on simulated new transaction data."""
    aml_engine = AMLEngine()
    fraud_engine = FraudEngine()
    
    new_aml_data, new_fraud_data = simulate_new_data()

    aml_predictions = aml_engine.predict(new_aml_data)
    fraud_predictions = fraud_engine.predict(new_fraud_data)

    if aml_predictions is not None:
        print("\nAML Predictions (first 5):")
        print(aml_predictions[['Is Laundering', 'aml_prediction']].head())
    
    if fraud_predictions is not None:
        print("\nFraud Predictions (first 5):")
        print(fraud_predictions[['Class', 'fraud_prediction']].head())

if __name__ == "__main__":
    predict_on_new_data()

