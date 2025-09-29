import pandas as pd
import os
from scripts.model_engine import AMLEngine, FraudEngine
from scripts.predict_new_data import simulate_new_data

def check_for_retraining(engine, new_data, threshold=0.01):
    """
    Simulates a check for concept drift or new cases to trigger retraining.
    """
    print(f"\nChecking retraining for {engine.name} engine...")
    
    if 'is_fraud' in new_data.columns:
        new_cases = new_data[new_data['is_fraud'] == 1]
    elif 'is_anomaly' in new_data.columns:
        new_cases = new_data[new_data['is_anomaly'] == 1]
    else:
        new_cases = pd.DataFrame() # No labels in the new data

    if not new_cases.empty and len(new_cases) / len(new_data) > threshold:
        print(f"Significant number of new cases detected ({len(new_cases)}). Retraining {engine.name} model.")
        
        historical_data_path = f'data/{engine.name}_historical_data.csv'
        
        # Load historical data if it exists
        if os.path.exists(historical_data_path):
            historical_data = pd.read_csv(historical_data_path)
            full_data = pd.concat([historical_data, new_data], ignore_index=True)
        else:
            full_data = new_data
        
        # Map old column names to new standard names for engine processing
        if engine.name == 'AML':
            full_data['Is Laundering'] = full_data['is_anomaly']
        elif engine.name == 'Fraud':
            full_data['Class'] = full_data['is_fraud']
        
        engine.train(full_data)
        full_data.to_csv(historical_data_path, index=False)
    else:
        print("Retraining not necessary based on new data.")

def retrain_all_models_with_new_data():
    """Retrains all models with a new batch of simulated data."""
    aml_engine = AMLEngine()
    fraud_engine = FraudEngine()
    
    new_aml_data, new_fraud_data = simulate_new_data(num_records=500, num_anomalies_aml=20, num_anomalies_fraud=10, seed=44)
    
    check_for_retraining(aml_engine, aml_engine.preprocess(new_aml_data))
    check_for_retraining(fraud_engine, fraud_engine.preprocess(new_fraud_data))

if __name__ == "__main__":
    retrain_all_models_with_new_data()

