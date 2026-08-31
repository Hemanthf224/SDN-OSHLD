import pandas as pd
import numpy as np
import os
import urllib.request
import zipfile
import tarfile

class DataLoader:
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def download_cicids2017(self):
        """
        Placeholder for CICIDS2017 download logic. 
        Because the dataset is > 2GB and hosted in multiple CSVs on Canadian Institute for Cybersecurity, 
        it's recommended to download manually from https://www.unb.ca/cic/datasets/ids-2017.html
        For now, this function will check if files exist.
        """
        dataset_path = os.path.join(self.raw_dir, 'CICIDS2017')
        if not os.path.exists(dataset_path):
            print(f"Please download CICIDS2017 manually to {dataset_path}")
            return False
        return True

    def download_nsl_kdd(self):
        """
        Downloads NSL-KDD dataset.
        """
        url = "http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip"  # Alternate URL might be needed
        # We will use a known Kaggle or openml link if needed, but for now we expect the user to have it or we can download a smaller version.
        dataset_path = os.path.join(self.raw_dir, 'NSL-KDD')
        if not os.path.exists(dataset_path):
            print(f"Please download NSL-KDD manually to {dataset_path}")
            return False
        return True

    def load_dataset(self, name):
        """
        Load processed dataset
        """
        path = os.path.join(self.processed_dir, f"{name}_processed.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        else:
            raise FileNotFoundError(f"Processed dataset not found at {path}")
