import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import logging

# --- Import risk data from the separate file ---
from aml_risk_data import RISK_CATEGORIES, assess_customer_risk

# Set up logging for better tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. AML Model Integration ---

def preprocess_data_for_prediction(data, scaler, column_names):
    """
    Cleans and preprocesses new data using a pre-trained scaler and column names.
    """
    data = data.drop(columns=['transaction_id'], errors='ignore')
    data = pd.get_dummies(data, columns=['transaction_type'], drop_first=True)
    
    for col in column_names:
        if col not in data.columns:
            data[col] = 0

    data = data.reindex(columns=column_names, fill_value=0)
    data = data.fillna(data.mean(numeric_only=True))
    scaled_data = scaler.transform(data)
    return scaled_data

def run_aml_engine(new_transactions_df, model, scaler, column_names, customer_data_df):
    """
    Uses the AML model to predict anomalies on new transactions.
    """
    if new_transactions_df.empty:
        return pd.DataFrame()
    
    # --- NEW: Enrich data with risk flags before prediction ---
    enriched_transactions_df = pd.merge(new_transactions_df.copy(), customer_data_df, on='customer_id', how='left')
    
    def get_risk_score_and_flags(row):
        customer_info = row.to_dict()
        transaction_info = pd.DataFrame([row])
        assessment = assess_customer_risk(customer_info, transaction_info)
        return assessment['risk_score'], ", ".join(assessment['triggered_flags'])

    enriched_transactions_df[['risk_score', 'triggered_flags']] = enriched_transactions_df.apply(
        lambda row: pd.Series(get_risk_score_and_flags(row)), axis=1
    )
    
    preprocessed_data_array = preprocess_data_for_prediction(enriched_transactions_df.copy(), scaler, column_names)
    preprocessed_data_df = pd.DataFrame(preprocessed_data_array, columns=column_names)
    
    predictions = model.predict(preprocessed_data_df)
    
    enriched_transactions_df['aml_flag'] = np.where(predictions == -1, 'Anomaly', 'Normal')
    
    return enriched_transactions_df

# --- 2. RAG Agent Setup ---

def setup_rag_agent():
    """
    Loads AML procedure documents into a LlamaIndex VectorStoreIndex using Ollama.
    """
    if not os.path.exists("docs"):
        st.error("RAG documents directory 'docs' not found. Please create it and add your AML procedures.")
        return None

    Settings.llm = Ollama(model="llama3", request_timeout=300.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    documents = SimpleDirectoryReader("docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    
    return query_engine

# --- 3. Streamlit Dashboard ---

def main():
    st.set_page_config(layout="wide")
    st.title("AML and Fraud Agent Dashboard (Ollama)")

    model_path = 'aml_model.joblib'
    scaler_path = 'aml_scaler.joblib'
    column_names_path = 'aml_column_names.joblib'
    customer_data_path = 'customer_data.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(column_names_path) or not os.path.exists(customer_data_path):
        st.error("AML model, scaler, column names, or customer data not found. Please run the training script and ensure customer_data.csv exists.")
        st.stop()

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        column_names = joblib.load(column_names_path)
        customer_data_df = pd.read_csv(customer_data_path)
    except Exception as e:
        st.error(f"Error loading model, scaler, column names, or customer data: {e}")
        st.stop()

    query_engine = setup_rag_agent()
    if query_engine is None:
        st.stop()

    st.header("Upload New Transactions")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if 'processed_df' not in st.session_state:
        st.session_state['processed_df'] = pd.DataFrame()

    if uploaded_file:
        transactions_df = pd.read_csv(uploaded_file)
        st.session_state['processed_df'] = run_aml_engine(transactions_df.copy(), model, scaler, column_names, customer_data_df)
        
    if not st.session_state['processed_df'].empty:
        st.header("AML Engine Results")
        processed_df = st.session_state['processed_df']
        anomalies_df = processed_df[processed_df['aml_flag'] == 'Anomaly']
        
        if not anomalies_df.empty:
            st.warning(f"**{len(anomalies_df)} potential anomalies detected!**")
            st.dataframe(anomalies_df)
            
            st.header("RAG Agent: Explain an Anomaly")
            selected_transaction = st.selectbox(
                "Select a transaction ID to investigate:",
                anomalies_df['transaction_id'].unique()
            )
            
            if selected_transaction:
                # --- MODIFICATION: Include risk flags in details for RAG ---
                transaction_details = anomalies_df[anomalies_df['transaction_id'] == selected_transaction].to_string()
                triggered_flags = anomalies_df[anomalies_df['transaction_id'] == selected_transaction]['triggered_flags'].iloc[0]
                
                query = f"""
                Based on the AML procedure documents, provide an analysis for this flagged transaction:
                Transaction Details:\n{transaction_details}\n
                Customer Risk Flags: {triggered_flags}\n
                Highlight the key risk factors and suggest the next steps for a compliance officer.
                """
                
                with st.spinner("Analyzing with RAG agent (Ollama)..."):
                    try:
                        rag_response = query_engine.query(query)
                        st.subheader(f"RAG Agent Analysis for Transaction {selected_transaction}")
                        
                        if rag_response.source_nodes:
                            source_texts = "\n\n".join([n.text for n in rag_response.source_nodes])
                            with st.expander("LLM Analysis"):
                                st.info(rag_response.response)
                            with st.expander("Supporting Document Excerpt", expanded=True):
                                st.code(source_texts, language='markdown')
                                st.markdown("---")
                                st.caption("This excerpt from `aml_procedures.txt` provides context for the analysis.")
                        else:
                            st.info(rag_response.response)
                        
                    except Exception as e:
                        st.error(f"Failed to get RAG response from Ollama: {e}")
        else:
            st.success("No anomalies detected in the uploaded transactions.")
    else:
        st.info("Please upload a CSV file to begin.")

if __name__ == "__main__":
    main()


