import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. AML Model Integration ---

# Re-define the preprocessing function to use in the app
def preprocess_data_for_prediction(data, scaler):
    """
    Cleans and preprocesses new data using a pre-trained scaler.
    """
    # Drop irrelevant columns if they exist
    data = data.drop(columns=['transaction_id'], errors='ignore')
    
    # One-hot encode categorical features (assumes same categories as training)
    data = pd.get_dummies(data, columns=['transaction_type'], drop_first=True)
    
    # Realign columns in case of missing transaction types in new data
    # Assumes training data had 'debit', 'credit', 'transfer'
    known_columns = ['customer_id', 'transaction_amount', 'transaction_type_debit', 'transaction_type_transfer']
    for col in known_columns:
        if col not in data.columns:
            data[col] = 0

    # Ensure consistent column order
    data = data.reindex(columns=known_columns, fill_value=0)

    # Handle any missing values
    data = data.fillna(data.mean(numeric_only=True))

    # Scale the numerical features using the pre-trained scaler
    scaled_data = scaler.transform(data)
    return scaled_data

def run_aml_engine(new_transactions_df, model, scaler):
    """
    Uses the AML model to predict anomalies on new transactions.
    """
    if new_transactions_df.empty:
        return pd.DataFrame()

    preprocessed_data_array = preprocess_data_for_prediction(new_transactions_df.copy(), scaler)
    
    
    # Get the column names from the scaler's original feature names
    # This assumes the scaler was trained on a DataFrame and saved with its feature names.
    # We will use the columns from the original training data for robustness.
    # To get this, we need to load the training data again or save the columns.
    
    # For demonstration, we will retrieve the column names.
    # In a real-world scenario, you might save the column names list
    # with your model artifacts for consistency.
    
    # --- IMPORTANT FIX STARTS HERE ---
    # Load a sample of the training data to get the original column names
    # You could also save this list as an artifact during training.
    train_data_sample = pd.read_csv('aml_transaction_data.csv')
    train_data_sample = train_data_sample.drop(columns=['transaction_id', 'is_laundering'], errors='ignore')
    train_data_sample = pd.get_dummies(train_data_sample, columns=['transaction_type'], drop_first=True)
    column_names = train_data_sample.columns

    # Convert the preprocessed NumPy array back to a DataFrame
    preprocessed_data_df = pd.DataFrame(preprocessed_data_array, columns=column_names)
    
    # Now, make the prediction using the DataFrame
    predictions = model.predict(preprocessed_data_df)
    
    # Interpret IsolationForest predictions: -1 is anomalous, 1 is normal
    new_transactions_df['aml_flag'] = np.where(predictions == -1, 'Anomaly', 'Normal')
    
    return new_transactions_df
# --- 2. RAG Agent Setup ---

def setup_rag_agent():
    """
    Loads AML procedure documents into a LlamaIndex VectorStoreIndex using Ollama.
    """
    if not os.path.exists("docs"):
        st.error("RAG documents directory 'docs' not found. Please create it and add your AML procedures.")
        return None

    # Configure LlamaIndex to use Ollama for the LLM
    Settings.llm = Ollama(model="llama3", request_timeout=300)
    
    # Use a local HuggingFace model for embeddings (avoids external API)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Load documents from the 'docs' directory
    documents = SimpleDirectoryReader("docs").load_data()
    
    # Create the vector store index
    index = VectorStoreIndex.from_documents(documents)
    
    # Create a query engine
    query_engine = index.as_query_engine()
    
    return query_engine

# --- 3. Streamlit Dashboard ---

def main():
    st.set_page_config(layout="wide")
    st.title("AML and Fraud Agent Dashboard (Ollama)")

    # Load the pre-trained AML model and scaler
    model_path = 'aml_model.joblib'
    scaler_path = 'aml_scaler.joblib' # Assume scaler was saved
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("AML model or scaler not found. Please run the training script and save the scaler.")
        st.stop()

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

    # Setup RAG agent
    query_engine = setup_rag_agent()
    if query_engine is None:
        st.stop()

    st.header("Upload New Transactions")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    transactions_df = pd.DataFrame()
    if uploaded_file:
        transactions_df = pd.read_csv(uploaded_file)
        
        # Add 'aml_flag' column, but keep normal transactions for display
        st.session_state['processed_df'] = run_aml_engine(transactions_df.copy(), model, scaler)
        
    if 'processed_df' in st.session_state:
        st.header("AML Engine Results")
        processed_df = st.session_state['processed_df']
        anomalies_df = processed_df[processed_df['aml_flag'] == 'Anomaly']
        
        if not anomalies_df.empty:
            st.warning(f"**{len(anomalies_df)} potential anomalies detected!**")
            st.dataframe(anomalies_df)
            
            # --- RAG Interaction for Anomalies ---
            st.header("RAG Agent: Explain an Anomaly")
            selected_transaction = st.selectbox(
                "Select a transaction ID to investigate:",
                anomalies_df['transaction_id'].unique()
            )
            
            if selected_transaction:
                transaction_details = anomalies_df[anomalies_df['transaction_id'] == selected_transaction].to_string()
                
                # Create a query for the agent
                query = f"""
                Based on the AML procedure documents, provide an analysis for this flagged transaction:
                Transaction Details:\n{transaction_details}\n
                Highlight the key risk factors and suggest the next steps for a compliance officer.
                """
                
                # Get response from RAG agent
                with st.spinner("Analyzing with RAG agent (Ollama)..."):
                    try:
                        rag_response = query_engine.query(query)
                        st.subheader(f"RAG Agent Analysis for Transaction {selected_transaction}")
                        st.info(rag_response.response)
                    except Exception as e:
                        st.error(f"Failed to get RAG response from Ollama: {e}")
        else:
            st.success("No anomalies detected in the uploaded transactions.")

if __name__ == "__main__":
    main()
