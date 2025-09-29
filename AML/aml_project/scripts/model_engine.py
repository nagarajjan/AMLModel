import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class ModelEngine:
    """Base class for a modular ML model engine."""
    def __init__(self, name, model_path):
        self.name = name
        self.model_path = os.path.join(MODEL_DIR, f"{model_path}.joblib")
        self.scaler_path = os.path.join(MODEL_DIR, f"{model_path}_scaler.joblib")
        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        """Loads a model from disk if it exists."""
        return joblib.load(self.model_path) if os.path.exists(self.model_path) else None

    def _load_scaler(self):
        """Loads a scaler from disk if it exists."""
        return joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else None

    def _save_model_artifacts(self, model, scaler):
        """Saves the trained model and scaler."""
        joblib.dump(model, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        print(f"{self.name} model and scaler saved.")

    def preprocess(self, df):
        """Placeholder for preprocessing logic."""
        raise NotImplementedError

    def train(self, df):
        """Trains the model on the provided data."""
        processed_df = self.preprocess(df)
        target_column = 'is_fraud' if 'is_fraud' in processed_df.columns else 'is_anomaly'
        
        # Handle cases where no anomalies exist to prevent errors
        if np.sum(processed_df[target_column] == 1) == 0:
            contamination = 0.01 # Fallback contamination if no anomalies in training set
        else:
            contamination = float(np.sum(processed_df[target_column] == 1) / len(processed_df))

        X = processed_df.drop(target_column, axis=1, errors='ignore')
        y = processed_df[target_column]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination, random_state=42)
        model.fit(X_scaled)
        
        self.model = model
        self.scaler = scaler
        self._save_model_artifacts(model, scaler)
        print(f"{self.name} model trained and saved.")

    def predict(self, new_transactions):
        """Predicts on new transaction data."""
        if not self.model or not self.scaler:
            print(f"{self.name} model not trained. Please train the model first.")
            return None
            
        processed_df = self.preprocess(new_transactions)
        
        X_predict = processed_df.drop('is_fraud' if 'is_fraud' in processed_df.columns else 'is_anomaly', axis=1, errors='ignore')
        
        X_predict_scaled = self.scaler.transform(X_predict)
        predictions = self.model.predict(X_predict_scaled)
        
        new_transactions[f'{self.name}_prediction'] = predictions
        return new_transactions

class AMLEngine(ModelEngine):
    """Engine for Anti-Money Laundering detection."""
    def __init__(self):
        super().__init__('AML', 'aml_model')

    def preprocess(self, df):
        """Specialized preprocessing for AML data."""
        df = df.copy()
        if 'Is Laundering' in df.columns:
            df['is_anomaly'] = df['Is Laundering']
            df = df.drop(columns=['Is Laundering'], errors='ignore')
        
        df = pd.get_dummies(df, columns=['FromBank', 'ToBank', 'type'], dummy_na=False, drop_first=True)
        
        if 'Timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['Timestamp'])
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['time_of_day'] = df['datetime'].dt.hour
            df = df.drop(columns=['Timestamp', 'datetime'], errors='ignore')
            
        df = df.drop(columns=['From', 'To'], errors='ignore')
        
        return df

class FraudEngine(ModelEngine):
    """Engine for general Fraud detection."""
    def __init__(self):
        super().__init__('Fraud', 'fraud_model')

    def preprocess(self, df):
        """Specialized preprocessing for Fraud data."""
        df = df.copy()
        if 'Class' in df.columns:
            df['is_fraud'] = df['Class']
            df = df.drop(columns=['Class'], errors='ignore')
        
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'], errors='ignore')
            
        return df

