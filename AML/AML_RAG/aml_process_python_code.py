import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import logging
import os

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Data Handling Functions ---

def load_data(file_path):
    """Loads transaction data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None

def preprocess_data(data):
    """Cleans and preprocesses the raw transaction data.
    
    Args:
        data (pd.DataFrame): The raw input transaction data.

    Returns:
        tuple: A tuple containing the preprocessed features (X), 
               the preprocessed target variable (y), and the scaler object.
    """
    logging.info("Starting data preprocessing.")
    
    # Drop irrelevant or redundant columns if they exist
    # Example: 'transaction_id' is often not useful for the model
    data = data.drop(columns=['transaction_id'], errors='ignore')

    # Convert categorical variables to numerical
    # Here, we assume 'transaction_type' is a categorical column.
    if 'transaction_type' in data.columns:
        data = pd.get_dummies(data, columns=['transaction_type'], drop_first=True)
        
    # Handle any missing values (for simplicity, we'll fill with the mean)
    data = data.fillna(data.mean(numeric_only=True))

    # Assume the target variable is named 'is_laundering'.
    if 'is_laundering' not in data.columns:
        # For unsupervised anomaly detection, there's no target variable
        X = data.copy()
        y = None
        logging.warning("No 'is_laundering' column found. Proceeding with unsupervised learning.")
    else:
        X = data.drop('is_laundering', axis=1)
        y = data['is_laundering']

    # Scale the numerical features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert back to DataFrame to keep column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    logging.info("Data preprocessing completed.")
    return X_scaled, y, scaler

def feature_engineer(data):
    """Creates new features from existing transaction data."""
    logging.info("Starting feature engineering.")
    df = data.copy()
    
    # Check if 'customer_id' and 'transaction_amount' columns exist
    if 'customer_id' in df.columns and 'transaction_amount' in df.columns:
        # Create a feature for transaction amount relative to the customer's average
        df['customer_avg_amount'] = df.groupby('customer_id')['transaction_amount'].transform('mean')
        df['is_high_value'] = (df['transaction_amount'] > (df['customer_avg_amount'] * 1.5)).astype(int)
    
    logging.info("Feature engineering completed.")
    return df

def save_model(model, file_path):
    """Saves the trained model to a file."""
    try:
        joblib.dump(model, file_path)
        logging.info(f"Model saved successfully to {file_path}.")
    except Exception as e:
        logging.error(f"Failed to save the model: {e}")

# --- 2. Model Management Functions ---

def train_or_retrain_model(data_path, model_path=None, scaler_path=None, test_size=0.2, random_state=42):
    """Creates a new model or retrains an existing one.
    
    Args:
        data_path (str): Path to the training data CSV file.
        model_path (str, optional): Path to a pre-trained model file (.joblib). 
                                    If None, a new model is trained. Defaults to None.
        scaler_path (str, optional): Path to save the fitted scaler. Defaults to None.
        test_size (float, optional): Proportion of the dataset to include in the test split.
        random_state (int, optional): Controls the shuffling applied to the data before splitting.
                                  
    Returns:
        object: The trained or retrained model.
    """
    logging.info("Starting model training process.")
    
    # Load and preprocess data
    data = load_data(data_path)
    if data is None:
        return None
    
    X, y, scaler = preprocess_data(data)
    
    if scaler_path:
        save_model(scaler, scaler_path)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Initialize or load model
    if model_path and os.path.exists(model_path):
        logging.info(f"Loading existing model from {model_path}.")
        try:
            model = joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            return None
        
        # Retrain the model on the new data
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
        
    # --- Evaluation ---
    logging.info("Evaluating model performance.")
    
    # For Isolation Forest, we predict anomalies (-1) vs normal (1)
    if y_test is not None:
        y_pred = model.fit_predict(X_test)
        # Convert IsolationForest output to binary labels (1 -> 0, -1 -> 1)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # We assume the test set has labels for evaluation purposes
        logging.info("\n" + classification_report(y_test, y_pred_binary, zero_division=0))
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")
        
    # Save the new or retrained model
    save_model(model, model_path if model_path else 'aml_model.joblib')
    
    return model

# --- Main Execution Block ---

if __name__ == "__main__":
    # Configuration paths
    MODEL_FILE = 'aml_model.joblib'
    SCALER_FILE = 'aml_scaler.joblib'
    TRAINING_DATA_FILE = 'aml_transaction_data.csv'
    NEW_TRANSACTION_DATA_FILE = 'new_aml_transactions.csv'
    
    # --- Step 1: Simulate data creation ---
    logging.info("Generating synthetic data for demonstration.")
    
    # Create synthetic data for training
    np.random.seed(42)
    data = pd.DataFrame({
        'transaction_id': range(10000),
        'customer_id': np.random.randint(1, 1000, 10000),
        'transaction_amount': np.random.lognormal(mean=7, sigma=1.5, size=10000),
        'transaction_type': np.random.choice(['debit', 'credit', 'transfer'], 10000),
        'is_laundering': np.random.choice([0, 1], 10000, p=[0.99, 0.01]) # 1% anomalies
    })
    data.to_csv(TRAINING_DATA_FILE, index=False)

    # Create some new data with potential anomalies
    new_data = pd.DataFrame({
        'transaction_id': range(10000, 10100),
        'customer_id': np.random.randint(1, 1000, 100),
        'transaction_amount': np.random.lognormal(mean=7, sigma=1.5, size=100),
        'transaction_type': np.random.choice(['debit', 'credit', 'transfer'], 100),
    })
    # Add a definite anomaly
    new_data.loc[50, 'transaction_amount'] = 50000  # Unusually high transaction
    new_data.to_csv(NEW_TRANSACTION_DATA_FILE, index=False)
    
    # --- Step 2: Initial Model Training and Saving Artifacts ---
    logging.info("--- Phase 1: Initial Training ---")
    initial_model = train_or_retrain_model(TRAINING_DATA_FILE, model_path=MODEL_FILE, scaler_path=SCALER_FILE)
    
    # --- Step 3: Retraining on New Data ---
    logging.info("\n--- Phase 2: Retraining with New Data ---")
    # For a real scenario, you'd combine new labeled data with the old.
    # Here, we demonstrate the retraining logic using the same file.
    retrained_model = train_or_retrain_model(TRAINING_DATA_FILE, model_path=MODEL_FILE, scaler_path=SCALER_FILE)
    
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        logging.info("Model and scaler saved. Ready for dashboard.")
    else:
        logging.error("Failed to save model and/or scaler.")

