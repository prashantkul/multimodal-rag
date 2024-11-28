from google.cloud import storage
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import List, Tuple, Any, Dict
from tqdm import tqdm
from io import BytesIO
import os
from collections import defaultdict
import threading

from config_manager import ConfigManager
from product import Product
from product import DataManager

class ImageManager:
    """
    Handles downloading product images and streaming them directly to Google Cloud Storage.
    """
    
    def __init__(self):
        """
        Initialize the ImageManager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_manager = ConfigManager()
        
        # Get other configurations as needed
        gcp_config = self.config_manager.get_gcp_config()
        self.max_workers = gcp_config.get('max_workers', 10)
        self.max_retries = gcp_config.get('max_retries', 3)
        self.timeout = gcp_config.get('timeout', 10)

         # Initialize GCS client
        self.storage_client = storage.Client(project=self.config_manager.project_id)
        self.bucket_name = self.config_manager.bucket_name
        self.products_prefix = self.config_manager.get_image_processing_config()['products_prefix']
        self.image_counter = 0  # Add counter for sequential naming

        # Initialize GCS client and get bucket
        storage_client = storage.Client()
        self.bucket = storage_client.bucket(self.bucket_name)
        

    def _process_single_image(self, product: Product, retry_count: int = 0) -> Tuple[str, bool, Any]:
        """
        Download a single image 
        
        Args:
            product: Product instance containing image information
            retry_count: Current retry attempt number
                
        Returns:
            Tuple of (product_name, success_status, gcs_path or error_message)
        """
        failed_products = []
        
        try:
            # Extract file extension from URL
            parsed_url = urlparse(product.image_url)
            ext = os.path.splitext(parsed_url.path)[1].lower()
            if not ext:
                ext = '.jpg'  # Default to jpg if no extension
            
            # Get next image number and format with leading zeros
            image_num = product.unique_id
            image_name = f"{image_num}{ext}"  # Will produce: image_000001.jpg
            
            gcs_path = f"{self.products_prefix}{image_name}"
            print(f"***  Processing {product.unique_id} to GCS path: {gcs_path} ***")
            
            # Download image to memory
            response = requests.get(product.image_url, timeout=self.timeout, stream=True)
            response.raise_for_status()
            
            # Create blob and upload directly from memory
            blob = self.bucket.blob(gcs_path)
            blob.upload_from_file(
                BytesIO(response.content),
                content_type=response.headers.get('content-type', 'image/jpeg')
            )
            
            return product.name, True, gcs_path
            
        except Exception as e:
            # if retry_count < self.max_retries:
            #     print(f"\nRetrying download for product {product.name}. Error: {str(e)}")
            #     return self._process_single_image(product, retry_count + 1)
            # else:
            #     error_msg = f"Failed to process image for product {product.name}: {str(e)}"
            #     return product.name, False, error_msg
            
            error_msg = f"Failed to process image for product {product.unique_id}: {str(e)}"
            return product.unique_id, False, error_msg  
    
    def process_images(self) -> Dict[str, Any]:
        """
        Process multiple images in parallel and update DataFrame with results.
        
        Args:
            df: DataFrame containing product information
            
        Returns:
            DataFrame with additional columns for GCS path and processing status
        """
        data_manager = DataManager()
        products = data_manager.get_all_products()
        #products = data_manager.get_sample_products(25)
        
        print(f"\nProcessing {len(products)} images...")
        print(f"GCS Bucket: {self.bucket_name}")
        print(f"Products prefix: {self.products_prefix}")
        
        # Track processing results
        results = defaultdict(list)
        success_count = 0
        error_count = 0
        
        
        # Process images in parallel with progress bar
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_single_image, product)
                for product in products
            ]
            
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Uploading images to GCS"
            ):
                try:
                    unique_id, success, result = future.result()
                    
                    if success:
                        success_count += 1
                        results['successful_products'].append(unique_id)
                        results['gcs_paths'].append(result)
                    else:
                        error_count += 1
                        results['failed_products'].append(unique_id)
                        results['errors'].append(f"{unique_id}: {result}")
                        
                except Exception as e:
                    error_count += 1
                    results['failed_products'].append("Unknown")
                    results['errors'].append(f"Unexpected error: {str(e)}")

        # Compile summary statistics
        summary = {
            'total_products': len(products),
            'successful_downloads': success_count,
            'failed_downloads': error_count,
            'success_rate': f"{(success_count / len(products) * 100):.1f}%",
            'successful_products': results['successful_products'],
            'failed_products': results['failed_products'],
            'errors': results['errors'],
            'gcs_paths': results['gcs_paths']
        }
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total products processed: {summary['total_products']}")
        print(f"Successful downloads: {summary['successful_downloads']}")
        print(f"Failed downloads: {summary['failed_downloads']}")
        print(f"Success rate: {summary['success_rate']}")
        
        if error_count > 0:
            print("\nErrors encountered:")
            for error in summary['errors']:
                print(f"- {error}")
        
        return summary

# Test
image_mgr = ImageManager()
image_mgr.process_images()
    