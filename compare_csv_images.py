from google.cloud import storage
import pandas as pd
from config_manager import ConfigManager
import os
from typing import List, Dict, Set, Tuple

class MismatchChecker:
    """Checks for mismatches between CSV unique IDs and image filenames."""
    
    def __init__(self):
        """Initialize the MismatchChecker."""
        self.config_manager = ConfigManager()
        self.storage_client = storage.Client(project=self.config_manager.project_id)
        self.bucket_name = self.config_manager.bucket_name
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
    def get_csv_ids(self) -> Set[str]:
        """Get all unique IDs from the CSV."""
        csv_path = self.config_manager.get_amazon_config()['csv_path']
        gcs_prefix = self.config_manager.get_amazon_config()['gcs_prefix']
        full_path = f"gs://{self.bucket_name}/{gcs_prefix}{csv_path}"
        
        print(f"Reading CSV from: {full_path}")
        df = pd.read_csv(full_path)
        return set(df['Uniq Id'].astype(str))
    
    def get_image_ids(self) -> Set[str]:
        """Get all image IDs from the GCS bucket."""
        images_prefix = self.config_manager.get_image_processing_config()['products_prefix']
        blobs = self.bucket.list_blobs(prefix=images_prefix)
        
        # Extract unique IDs from image filenames
        image_ids = set()
        for blob in blobs:
            # Get filename without path and extension
            filename = os.path.basename(blob.name)
            image_id = os.path.splitext(filename)[0]
            image_ids.add(image_id)
            
        return image_ids

    def check_mismatches(self) -> Dict[str, List[str]]:
        """
        Check for mismatches between CSV IDs and image IDs.
        
        Returns:
            Dictionary containing:
            - 'missing_images': List of IDs in CSV but no corresponding image
            - 'orphaned_images': List of image IDs with no corresponding CSV entry
        """
        # Get IDs from both sources
        csv_ids = self.get_csv_ids()
        image_ids = self.get_image_ids()
        
        # Find mismatches
        missing_images = csv_ids - image_ids  # In CSV but no image
        orphaned_images = image_ids - csv_ids  # Have image but not in CSV
        
        results = {
            'missing_images': sorted(list(missing_images)),
            'orphaned_images': sorted(list(orphaned_images))
        }
        
        # Print summary
        print("\nMismatch Summary:")
        print(f"Total products in CSV: {len(csv_ids):,}")
        print(f"Total images found: {len(image_ids):,}")
        print(f"Products missing images: {len(missing_images):,}")
        print(f"Orphaned images: {len(orphaned_images):,}")
        
        # Print some examples if mismatches found
        if missing_images:
            print("\nExample products missing images (up to 5):")
            for id in list(missing_images)[:5]:
                print(f"- {id}")
                
        if orphaned_images:
            print("\nExample orphaned images (up to 5):")
            for id in list(orphaned_images)[:5]:
                print(f"- {id}")
        
        return results

def main():
    checker = MismatchChecker()
    mismatches = checker.check_mismatches()
    
    # You can use the results for further processing
    missing_images = mismatches['missing_images']
    orphaned_images = mismatches['orphaned_images']
    
    print(f"\nFound {len(missing_images)} products without images")
    print(f"Found {len(orphaned_images)} orphaned images")
    for id in orphaned_images:
        print(f"- {id}")

if __name__ == "__main__":
    main()