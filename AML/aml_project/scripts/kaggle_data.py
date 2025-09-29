import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_kaggle_data(dataset_name, path='data', unzip=True):
    """Downloads and unzips a dataset from Kaggle."""
    print(f"Downloading dataset: {dataset_name}...")
    api = KaggleApi()
    api.authenticate()
    
    # Create directory if it doesn't exist
    dataset_path = os.path.join(path, dataset_name.split('/')[-1])
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        api.dataset_download_files(dataset_name, path=dataset_path, unzip=unzip)
        print("Download complete.")
    else:
        print("Data already exists. Skipping download.")
        
    return dataset_path

