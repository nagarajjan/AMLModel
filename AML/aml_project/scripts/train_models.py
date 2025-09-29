import pandas as pd
import os
from scripts.model_engine import AMLEngine, FraudEngine
from scripts.kaggle_data import download_kaggle_data

KAGGLE_DATASET_AML = 'ealtman2019/ibm-transactions-for-anti-money-laundering-aml'
KAGGLE_DATASET_FRAUD = 'ealtman2019/credit-card-transactions'

def train_all_models():
    """Downloads data and trains all model engines."""
    # Download data from Kaggle
    aml_data_path = download_kaggle_data(KAGGLE_DATASET_AML, path='data/aml')
    fraud_data_path = download_kaggle_data(KAGGLE_DATASET_FRAUD, path='data/fraud')
    
    # Load and combine AML training data
    aml_files = [f for f in os.listdir(aml_data_path) if f.startswith('HI')]
    aml_train_data = pd.concat([pd.read_csv(os.path.join(aml_data_path, f)) for f in aml_files], ignore_index=True)
    
    # Load fraud training data
    fraud_train_data = pd.read_csv(os.path.join(fraud_data_path, 'creditcard.csv'))

    # Initialize and train model engines
    aml_engine = AMLEngine()
    fraud_engine = FraudEngine()
    
    aml_engine.train(aml_train_data)
    fraud_engine.train(fraud_train_data)

if __name__ == "__main__":
    train_all_models()

