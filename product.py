from dataclasses import dataclass
from typing import Optional
import pandas as pd

from config_manager import ConfigManager

@dataclass
class Product:
    """Represents a product with its attributes."""
    unique_id: str
    name: str
    category: str
    image_url: str
    product_url: str
    about_product: str
    selling_price: Optional[float] = None
    gcs_path: Optional[str] = None
    image_processed: bool = False
    processing_error: Optional[str] = None

class DataManager:
    """Manages product data from CSV."""
    
    def __init__(self):
        """
        Initialize DataManager with CSV path.
        
        Args:
            csv_path: Path to the input CSV file
        """
        # read the file name from config_manager and read the actual file from GCS bucket
        
        config_manager = ConfigManager()
        csv_path = config_manager.get_amazon_config()['csv_path']
        bucket = config_manager.get_gcp_config()['bucket_name']
        prefix = config_manager.get_amazon_config()['gcs_prefix']
        gcs_path = "gs://" + bucket + "/" + prefix + csv_path
        print(gcs_path)
        
        # Read CSV and convert to list of Product objects
        df = pd.read_csv(gcs_path)
        # Function to format column names
        def format_column_name(column_name):
            return column_name.lower().replace(' ', '_')

        # Apply the function to column names
        df = df.rename(columns=format_column_name)
        
        print("Read the products CSV successfully, shape of the df: ", df.shape)
     
        # Convert DataFrame rows to Product objects
        self.products = [
            Product(
                unique_id=str(row['uniq_id']),
                name=row['product_name'],
                category=row['category'],
                image_url=row['image'],
                product_url=row['product_url'],
                about_product=row['about_product'],
                selling_price=row['selling_price']
            )
            for _, row in df.iterrows()
        ]
        
        print(f"Loaded {len(self.products):,} products")
    
    def get_all_products(self) -> list[Product]:
        """Return all products."""
        return self.products
    
    def get_sample_products(self, n: int = 5) -> list[Product]:
        """Return sample of n products."""
        return self.products[:n]
    

    
# # Test
# dataManager = DataManager()
# print(dataManager.get_sample_products())