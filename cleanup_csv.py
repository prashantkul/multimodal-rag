from google.cloud import storage
import pandas as pd
from config_manager import ConfigManager
from product import Product, DataManager
from typing import List

class CSVCleaner:
    """Handles cleaning and updating product CSV based on failed product IDs."""
    
    def __init__(self):
        """Initialize the CSVCleaner with configuration."""
        self.config_manager = ConfigManager()
        self.storage_client = storage.Client(project=self.config_manager.project_id)
        self.bucket_name = self.config_manager.bucket_name
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
    def clean_csv(self, failed_ids: List[str]):
        """
        Remove failed products from CSV and update in GCS.
        
        Args:
            failed_ids: List of product IDs that failed image download
        """
        # Get CSV path from config
        csv_path = self.config_manager.get_amazon_config()['csv_path']
        gcs_prefix = self.config_manager.get_amazon_config()['gcs_prefix']
        full_path = f"gs://{self.bucket_name}/{gcs_prefix}{csv_path}"
        
        print(f"Reading CSV from: {full_path}")
        
        # Read the original CSV
        df = pd.read_csv(full_path)
        original_count = len(df)
        
        # Convert failed_ids to set for faster lookup
        failed_ids_set = set(failed_ids)
        
        # Remove failed products
        df = df[~df['Uniq Id'].astype(str).isin(failed_ids_set)]
        remaining_count = len(df)
        
        # Save updated CSV
        new_csv_name = csv_path.replace('.csv', '_cleaned_2.csv')
        new_full_path = f"gs://{self.bucket_name}/{gcs_prefix}{new_csv_name}"
        
        print("\nSaving cleaned CSV...")
        df.to_csv(new_full_path, index=False)
        
        # Print summary
        print("\nCleaning Summary:")
        print(f"Original product count: {original_count:,}")
        print(f"Failed products removed: {len(failed_ids):,}")
        print(f"Remaining products: {remaining_count:,}")
        print(f"Cleaned CSV saved to: {new_full_path}")
        
        return new_full_path

def main():
    # Example usage - replace this with your actual list of failed IDs
    failed_ids = [
            "52be5df83e35e8eead95bf914d24f020",
            "a92c3db26324a3abf7a46b82547d7632",
            "39d32b78f29fef8f0e0747189d4dfa91",
            "b12bd18a1dc3cbdfe41d604515d020d5",
            "8b72c9f54570aa49c213e5be3b9f2e86",
            "6308fe7ee781814a193e67bd8082d207",
            "c59ca5ebebc20fada333b978335bb5d0",
            "4e802668aed7b697b45d19703f92041a",
            "9bfa5338e720a3181c8213f032bb3726",
            "7d7609a6f2f7c252c3a0623100fbab4b",
            "aee348bac22aea518c5077dc9f9aafe7",
            "17fded6956e4129235a9dc5cd8db2b79",
            "1e27b2f4b27de88cc3e16e6c170e1699",
            "b39704af4d88eff0de3477630ba0d50c",
            "0cf32a8ed805be067c473c73c505407c",
            "9289171a3174690545cbb240b46078bf",
            "e7872d9a4af150ad97862455f2d14a69",
            "e086ff1f756e3e72dfd5295fd07c2310",
            "bd710303f4b5041c7b93aecad119d290",
            "7b52ded35bb9b86c3d9d4007912c9715",
            "88b796e1eab8ffd6a66702d49ce4b909",
            "19f71de4bae039c0bf512d71ab620134",
            "8628e1160dadbfe3f495f7cc13427b43",
            "7883289ad2435c222672d22faac5a59f",
            "59e7639b9ccde8f751782308d0a3b47d",
            "ba9e1b28f7fe60782980e73d8b388f34",
            "fb63e472cf58401f524d8ebef0e0d35f",
            "1cf80443b4b002e20b02a6a636210e35",
            "c272798cb147d374ed267136f1ae5ea1",
            "18ff0eeeaafa9f1e070a8f6be197998f",
            "9351e0e4a751fc75b9367e1ebc7d193e"
    ]
    
    cleaner = CSVCleaner()
    new_csv_path = cleaner.clean_csv(failed_ids)

if __name__ == "__main__":
    main()