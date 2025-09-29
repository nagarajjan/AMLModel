import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
from generate_scenarios import create_demo_data_files, create_scenario_file

# --- Step 1: Data Preparation ---
def load_and_preprocess_data(file_path):
    """
    Loads and prepares data from a CSV file, returning both the original and preprocessed data.
    """
    if not os.path.exists(file_path):
        print(f"Data file not found at {file_path}")
        return None, None

    df = pd.read_csv(file_path)
    original_df = df.copy() # Keep a copy of the original data for reporting
    
    # Handle optional columns
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Preprocessing for the Isolation Forest model
    features_to_model = df.select_dtypes(include=np.number).columns.tolist()
    
    # Feature engineering (example): transaction amount squared
    if 'transaction_amount' in features_to_model:
        df['amount_squared'] = df['transaction_amount'] ** 2
        features_to_model.append('amount_squared')
    
    # Fill missing values
    for col in features_to_model:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return original_df, df[features_to_model]

# --- Step 2: AML Model Management ---
class AMLModelManager:
    def __init__(self, model_dir='models', output_dir='output'):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.model_path = os.path.join(self.model_dir, 'aml_model.joblib')
        self.model = None
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def train_and_test(self, training_data, test_data):
        """
        Trains a new Isolation Forest model and evaluates it.
        """
        print("\nTraining a new Isolation Forest model...")
        self.model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
        self.model.fit(training_data)
        
        # Evaluate model performance on the test set
        test_predictions = self.model.predict(test_data)
        num_anomalies_test = np.sum(test_predictions == -1)
        print(f"Initial test: Detected {num_anomalies_test} anomalies in the test set.")
        
        self._save_model()
        print("New model trained, tested, and saved.")

    def retrain_model(self, new_training_data):
        """
        Retrains the model with new data.
        """
        if not self.model:
            print("No existing model to retrain. Loading or training a new one.")
            if not self.load_model():
                print("Could not load a model. Training a new model on the provided data.")
                train_data, test_data = train_test_split(new_training_data, test_size=0.2, random_state=42)
                self.train_and_test(train_data, test_data)
                return

        print("Retraining existing model with new data...")
        self.model.fit(new_training_data)
        self._save_model()
        print("Model retrained and saved.")

    def _save_model(self):
        """Saves the trained model to the specified path."""
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads the trained model from the specified path."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print("Model loaded successfully.")
            return True
        else:
            print("No existing model found.")
            return False

    def predict_and_report(self, original_data, preprocessed_data, report_name):
        """
        Uses the loaded model to predict anomalies and saves a report to a CSV file.
        Also provides insights based on scenario information if available.
        """
        if self.model:
            predictions = self.model.predict(preprocessed_data)
            
            # Combine original data with predictions
            original_data['is_anomaly'] = predictions == -1
            
            # Create a timestamped file name
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            report_path = os.path.join(self.output_dir, f"{report_name}_{timestamp}.csv")
            
            original_data.to_csv(report_path, index=False)
            
            num_anomalies = np.sum(predictions == -1)
            print(f"\nGenerated report: Detected {num_anomalies} anomalies. Saved to {report_path}")
            
            # --- Insights Section ---
            print("\n--- INSIGHTS ---")
            if 'scenario_type' in original_data.columns:
                scenario_anomalies = original_data[original_data['is_anomaly'] == True]
                for scenario_type in scenario_anomalies['scenario_type'].unique():
                    scenario_count = len(scenario_anomalies[scenario_anomalies['scenario_type'] == scenario_type])
                    print(f"  - Model detected {scenario_count} transactions matching scenario type: '{scenario_type}'.")
            else:
                print("  - No specific scenario information available for insights.")
            
            return original_data
        else:
            raise ValueError("No model is loaded. Train or load a model first.")

# --- Main AML Process Workflow ---
if __name__ == '__main__':
    # Step 1: Generate the CSV files and folder structure
    create_demo_data_files()
    create_scenario_file()

    # Step 2: First run - Initial training and testing
    print("\n--- FIRST RUN: Initial training and testing ---")
    model_manager = AMLModelManager()
    
    # Load and process initial data
    original_initial_train, initial_train_data = load_and_preprocess_data('data/initial_train.csv')
    original_initial_test, initial_test_data = load_and_preprocess_data('data/initial_test.csv')
    
    # Train and test the model
    model_manager.train_and_test(initial_train_data, initial_test_data)
    
    # Generate initial prediction report
    model_manager.predict_and_report(original_initial_test, initial_test_data, "initial_test_report")

    # Step 3: Retraining run - Simulate new data arrival
    print("\n--- RETRAINING RUN: Retraining with new data ---")
    
    # Load and process new data for retraining and holdout
    original_new_train, new_train_data = load_and_preprocess_data('data/new_data.csv')
    original_new_holdout, new_holdout_data = load_and_preprocess_data('data/new_data_holdout.csv')
    
    # Retrain the model
    model_manager.retrain_model(new_train_data)
    
    # Generate a report for the new holdout data
    model_manager.predict_and_report(original_new_holdout, new_holdout_data, "retrained_model_holdout_report")

    # Step 4: Scenario run - Test the model against a specific set of AML/Fraud scenarios
    print("\n--- SCENARIO RUN: Testing against predefined AML scenarios ---")
    
    original_scenario_data, preprocessed_scenario_data = load_and_preprocess_data('data/aml_scenario.csv')
    model_manager.predict_and_report(original_scenario_data, preprocessed_scenario_data, "aml_scenario_report")
