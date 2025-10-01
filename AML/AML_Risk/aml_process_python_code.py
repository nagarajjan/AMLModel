import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import logging
import os

# --- Import risk data from the separate file ---
from aml_risk_data import RISK_CATEGORIES, assess_customer_risk

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Data Handling Functions ---

def load_data(file_path):
    """Loads transaction data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None

def enrich_data_with_risk_flags(transactions_df, customer_data_df):
    """Joins transaction data with customer data and generates risk flags."""
    logging.info("Enriching data with customer risk information.")
    
    # Merge transaction data with customer data
    merged_df = pd.merge(transactions_df, customer_data_df, on='customer_id', how='left')

    # Apply risk assessment to each row
    def get_risk_score_and_flags(row):
        customer_info = row.to_dict()
        transaction_info = pd.DataFrame([row])
        assessment = assess_customer_risk(customer_info, transaction_info)
        return assessment['risk_score'], ", ".join(assessment['triggered_flags'])

    merged_df[['risk_score', 'triggered_flags']] = merged_df.apply(
        lambda row: pd.Series(get_risk_score_and_flags(row)), axis=1
    )
    
    return merged_df

def preprocess_data(data):
    """Cleans and preprocesses the raw transaction data."""
    logging.info("Starting data preprocessing.")
    
    # Drop columns not needed for the model
    data = data.drop(columns=['transaction_id', 'triggered_flags'], errors='ignore')
    
    # Drop customer-specific columns that might have caused the issue if not handled
    data = data.drop(columns=['country', 'industry', 'is_pep', 'complex_ownership_score'], errors='ignore')

    # Fill NaN values to ensure they don't appear in the assessment function
    data = data.fillna(data.mean(numeric_only=True))
    
    if 'transaction_type' in data.columns:
        data = pd.get_dummies(data, columns=['transaction_type'], drop_first=True)
        
    if 'is_laundering' not in data.columns:
        X = data.copy()
        y = None
        logging.warning("No 'is_laundering' column found. Proceeding with unsupervised learning.")
    else:
        X = data.drop('is_laundering', axis=1)
        y = data['is_laundering']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    logging.info("Data preprocessing completed.")
    return X_scaled, y, scaler

def save_model(model, file_path):
    """Saves the trained model to a file."""
    try:
        joblib.dump(model, file_path)
        logging.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")

# --- 2. Model Management Functions ---

def train_or_retrain_model(transactions_df, customer_df, model_path=None, scaler_path=None, column_names_path=None, test_size=0.2, random_state=42):
    """Creates a new model or retrains an existing one."""
    logging.info("Starting model training process.")
    
    if transactions_df is None or customer_df is None:
        return None
    
    # Add a check to convert boolean strings to actual booleans
    customer_df['is_pep'] = customer_df['is_pep'].astype(str).str.upper() == 'TRUE'

    enriched_df = enrich_data_with_risk_flags(transactions_df, customer_df)
    X, y, scaler = preprocess_data(enriched_df)
    
    if scaler_path:
        save_model(scaler, scaler_path)
        save_model(X.columns.tolist(), column_names_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading existing model from {model_path}.")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return None
        
        if hasattr(model, 'fit'):
            model.fit(X_train, y_train)
            logging.info("Model retrained successfully on new data.")
        else:
            logging.error("Loaded model does not have a 'fit' method.")
            return None
    else:
        logging.info("No existing model found. Training a new Isolation Forest model.")
        model = IsolationForest(contamination='auto', random_state=random_state)
        model.fit(X_train)
        
    logging.info("Evaluating model performance.")
    if y_test is not None:
        y_pred = model.fit_predict(X_test)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        logging.info("\n" + classification_report(y_test, y_pred_binary, zero_division=0))
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")
        
    save_model(model, model_path if model_path else 'aml_model.joblib')
    
    return model

# --- Main Execution Block ---

if __name__ == "__main__":
    MODEL_FILE = 'aml_model.joblib'
    SCALER_FILE = 'aml_scaler.joblib'
    COLUMN_NAMES_FILE = 'aml_column_names.joblib'
    TRAINING_DATA_FILE = 'aml_transaction_data.csv'
    CUSTOMER_DATA_FILE = 'customer_data.csv'
    
    # --- Step 1: Simulate data creation ---
    logging.info("Generating synthetic data for demonstration.")
    
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_id': range(10000),
        'customer_id': np.random.randint(1, 1000, 10000),
        'transaction_amount': np.random.lognormal(mean=7, sigma=1.5, size=10000),
        'transaction_type': np.random.choice(['debit', 'credit', 'transfer'], 10000),
        'is_laundering': np.random.choice([0, 1], 10000, p=[0.99, 0.01])
    })
    data.to_csv(TRAINING_DATA_FILE, index=False)
    
    if not os.path.exists(CUSTOMER_DATA_FILE):
        logging.error(f"Customer data file '{CUSTOMER_DATA_FILE}' not found. Please create it.")
        exit()
    
    # --- Step 2: Initial Model Training and Saving Artifacts ---
    logging.info("--- Phase 1: Initial Training ---")
    
    transactions_df = load_data(TRAINING_DATA_FILE)
    customer_df = load_data(CUSTOMER_DATA_FILE)
    
    if transactions_df is not None and customer_df is not None:
        train_or_retrain_model(transactions_df, customer_df, model_path=MODEL_FILE, scaler_path=SCALER_FILE, column_names_path=COLUMN_NAMES_FILE)
    
        logging.info("\n--- Phase 2: Retraining with New Data (example) ---")
        train_or_retrain_model(transactions_df, customer_df, model_path=MODEL_FILE, scaler_path=SCALER_FILE, column_names_path=COLUMN_NAMES_FILE)
    
        if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(COLUMN_NAMES_FILE):
            logging.info("Model, scaler, and column names saved. Ready for dashboard.")
        else:
            logging.error("Failed to save all necessary model artifacts.")
    else:
        logging.error("Failed to load necessary data for training.")

