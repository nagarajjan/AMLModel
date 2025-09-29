import os
from scripts.train_models import train_all_models
from scripts.predict_new_data import predict_on_new_data
from scripts.retrain_models import retrain_all_models_with_new_data

import sys
print(sys.path)

def main():
    """Main workflow to run the AML and Fraud detection process."""
    # Step 1: Initial setup - Train models if they don't exist
    if not os.path.exists('models/aml_model.joblib') or not os.path.exists('models/fraud_model.joblib'):
        print("--- Running initial model training ---")
        train_all_models()
    else:
        print("--- Existing models found. Skipping initial training ---")
    
    # Step 2: Make predictions on new data
    print("\n--- Running predictions on new data ---")
    predict_on_new_data()
    
    # Step 3: Check for retraining and trigger if necessary
    print("\n--- Checking and running retraining ---")
    retrain_all_models_with_new_data()

if __name__ == "__main__":
    main()

