import os

# Define the content of the corrected aml_risk_data.py file
file_content = """
import pandas as pd

# Define data structures for AML risk categories and red flags
RISK_CATEGORIES = {
    "pep": {
        "description": "Politically Exposed Person (PEP)",
        "customer_data_red_flags": [
            "Matches PEP screening database",
            "Known association with a PEP",
            "High public profile",
            "Reputation for corruption",
        ],
        "transaction_data_red_flags": [
            "Transactions with government entities",
            "Large, unexplained transfers",
            "Frequent international transfers to tax havens",
        ]
    },
    "high_risk_country": {
        "description": "Customer from or with transactions in a high-risk country",
        "customer_data_red_flags": [
            "Residency in a high-risk country (e.g., on FATF gray/black list)",
            "Citizenship in a high-risk country",
            "Frequent travel to high-risk countries",
        ],
        "transaction_data_red_flags": [
            "Transactions involving high-risk jurisdictions (e.g., sanctioned countries)",
            "Incoming funds from high-risk countries",
            "Frequent transactions with no apparent business purpose in high-risk countries",
        ]
    },
    "cash_intensive_business": {
        "description": "Cash-intensive business (e.g., restaurants, car washes)",
        "customer_data_red_flags": [
            "Industry type is cash-intensive",
            "Significant portion of revenue is cash",
            "Frequent cash deposits just below reporting thresholds",
        ],
        "transaction_data_red_flags": [
            "Frequent, large cash deposits",
            "Structure of deposits to avoid reporting thresholds",
            "Deposits far exceeding known business turnover",
            "Use of cash deposits to fund wire transfers",
        ]
    },
    "complex_ownership": {
        "description": "Customer with complex or opaque ownership structures",
        "customer_data_red_flags": [
            "Unclear or anonymous beneficial ownership",
            "Use of shell corporations or offshore entities",
            "Numerous layers of ownership",
            "Directors or owners residing in high-risk jurisdictions",
        ],
        "transaction_data_red_flags": [
            "Complex web of transactions between related entities",
            "Transactions with no clear business purpose between related parties",
            "Sudden changes in ownership or transaction patterns",
        ]
    }
}


def assess_customer_risk(customer_data, transaction_data):
    \"\"\"
    Assesses a customer's risk based on defined categories and red flags.

    Args:
        customer_data (dict): A dictionary of customer information.
        transaction_data (pd.DataFrame): A DataFrame of the customer's transactions.

    Returns:
        dict: A risk assessment with a score and a list of triggered red flags.
    \"\"\"
    risk_score = 0
    triggered_flags = []

    # Check for PEP status
    if customer_data.get("is_pep", False) == True:  # Changed to match boolean type
        risk_score += 5
        triggered_flags.append("PEP status detected.")
        triggered_flags.extend(RISK_CATEGORIES["pep"]["transaction_data_red_flags"])

    # Check for high-risk country
    if customer_data.get("country") in ["HighRiskCountryA", "HighRiskCountryB", "HighRiskCountryC"]:
        risk_score += 3
        triggered_flags.append(f"Involvement with high-risk country: {customer_data.get('country')}.")
        triggered_flags.extend(RISK_CATEGORIES["high_risk_country"]["transaction_data_red_flags"])
    
    # Check for cash-intensive business
    customer_industry = str(customer_data.get("industry", "")).lower()
    if customer_industry in ["restaurant", "auto dealer"]:
        risk_score += 2
        triggered_flags.append("Customer is from a cash-intensive industry.")
        triggered_flags.extend(RISK_CATEGORIES["cash_intensive_business"]["transaction_data_red_flags"])
        
    # Check for complex ownership
    if float(customer_data.get("complex_ownership_score", 0)) > 0.7:
        risk_score += 4
        triggered_flags.append("Complex ownership structure detected.")
        triggered_flags.extend(RISK_CATEGORIES["complex_ownership"]["transaction_data_red_flags"])

    # Check transaction data for red flags
    if "transfer" in transaction_data["transaction_type"].values and \\
       (transaction_data["transaction_amount"] > 10000).any():
        risk_score += 2
        triggered_flags.append("High-value transfer detected.")

    return {
        "risk_score": risk_score,
        "triggered_flags": list(set(triggered_flags))
    }
"""

# Create and write to the file
file_name = "aml_risk_data.py"

try:
    with open(file_name, "w") as f:
        f.write(file_content)
    print(f"File '{file_name}' created successfully with fixes.")
except Exception as e:
    print(f"An error occurred while creating the file: {e}")
