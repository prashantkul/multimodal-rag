import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Union, Tuple
from io import BytesIO
from google.cloud import storage
import numpy as np
from tqdm import tqdm

from config_manager import ConfigManager
from product import Product, DataManager

class EmbeddingManager:
    """Manages the generation and storage of CLIP embeddings for products."""
    
    def __init__(self):
        """Initialize the embedding manager with CLIP model and processor."""
        self.config_manager = ConfigManager()
        self.storage_client = storage.Client(project=self.config_manager.project_id)
        self.bucket = self.storage_client.bucket(self.config_manager.bucket_name)
        
        # Load CLIP model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        
        # Set embedding dimension
        self.embedding_dim = 512  # CLIP's default embedding dimension
        
    def _load_image_from_gcs(self, unique_id: str) -> Image.Image:
        """Load image from Google Cloud Storage using product's unique_id."""
        products_prefix = self.config_manager.get_image_processing_config()['products_prefix']
        # Try different possible extensions
        for ext in ['.jpg', '.jpeg', '.png']:
            try:
                blob_path = f"{products_prefix}{unique_id}{ext}"
                blob = self.bucket.blob(blob_path)
                image_bytes = blob.download_as_bytes()
                return Image.open(BytesIO(image_bytes))
            except Exception as e:
                continue
        raise Exception(f"Could not find image for product {unique_id}")
    
    def generate_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Generate embedding for a single image using CLIP."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            image_embedding = image_features.cpu().numpy()
            image_embedding = image_embedding / np.linalg.norm(image_embedding)
            
            return image_embedding.squeeze()

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text using CLIP."""
        with torch.no_grad():
            inputs = self.processor(text=text, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            text_features = self.model.get_text_features(**inputs)
            
            # Normalize embedding
            text_embedding = text_features.cpu().numpy()
            text_embedding = text_embedding / np.linalg.norm(text_embedding)
            
            return text_embedding.squeeze()

    def _prepare_text_content(self, product: Product) -> str:
        """Prepare text content from product name and description."""
        # Combine product name and about_product
        text_content = f"{product.name}. {product.about_product}"
        
        # Clean up text (remove multiple spaces, newlines, etc.)
        text_content = " ".join(text_content.split())
        
        return text_content

    def generate_product_embeddings(self, product: Product) -> Dict[str, Union[str, np.ndarray]]:
        """Generate embeddings for both image and text content of a product."""
        try:
            # Load image using unique_id
            image = self._load_image_from_gcs(product.unique_id)
            image_embedding = self.generate_image_embedding(image)
            
            # Generate text embedding using product name and description
            text_content = self._prepare_text_content(product)
            text_embedding = self.generate_text_embedding(text_content)
            
            return {
                'unique_id': product.unique_id,
                'image_embedding': image_embedding,
                'text_embedding': text_embedding,
                'product_name': product.name,
                'text_content': text_content, # Store the actual text used for embedding
                'category': product.category
            }
            
        except Exception as e:
            print(f"Error processing product {product.unique_id}: {str(e)}")
            return None

    def process_batch(self, products: List[Product], batch_size: int = 32) -> List[Dict[str, Union[str, np.ndarray]]]:
        """Process a batch of products to generate embeddings."""
        all_embeddings = []
        failed_products = []
        
        for i in tqdm(range(0, len(products), batch_size), desc="Processing batches"):
            batch = products[i:i + batch_size]
            batch_embeddings = []
            
            for product in batch:
                embeddings = self.generate_product_embeddings(product)
                if embeddings:
                    batch_embeddings.append(embeddings)
                else:
                    failed_products.append(product.unique_id)
            
            all_embeddings.extend(batch_embeddings)
        
        # Print summary
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {len(all_embeddings)} products")
        print(f"Failed to process: {len(failed_products)} products")
        
        if failed_products:
            print("\nFailed product IDs:")
            for id in failed_products[:10]:  # Show first 10 failures
                print(f"- {id}")
            if len(failed_products) > 10:
                print(f"...and {len(failed_products) - 10} more")
        
        return all_embeddings

    def find_similar_products(self, 
                            query_embedding: np.ndarray, 
                            product_embeddings: List[Dict[str, Union[str, np.ndarray]]], 
                            k: int = 5,
                            modality: str = 'text') -> List[Tuple[str, float, str, str]]:
        """
        Find k most similar products based on embedding similarity.
        
        Returns:
            List of tuples containing (unique_id, similarity_score, product_name, text_content)
        """
        similarities = []
        embedding_key = f"{modality}_embedding"
        
        for prod_embed in product_embeddings:
            if embedding_key in prod_embed and prod_embed[embedding_key] is not None:
                similarity = np.dot(query_embedding, prod_embed[embedding_key])
                similarities.append((
                    prod_embed['unique_id'],
                    similarity,
                    prod_embed['product_name'],
                    prod_embed['text_content']
                ))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

def main():
    """Test the embedding generation pipeline."""
    data_manager = DataManager()
    embedding_manager = EmbeddingManager()
    
    # Get sample products
    products = data_manager.get_sample_products(4)
    print(f"Processing {len(products)} products...")
    
    # Generate embeddings
    embeddings = embedding_manager.process_batch(products, batch_size=5)
    print(f"Generated embeddings for {len(embeddings)} products")
    
    # Test similarity search
    if embeddings:
        # Test text query
        query_text = "smartphone with good camera"
        query_embedding = embedding_manager.generate_text_embedding(query_text)
        similar_products = embedding_manager.find_similar_products(
            query_embedding, 
            embeddings,
            k=3,
            modality='text'
        )
        
        print("\nMost similar products to query:", query_text)
        for unique_id, similarity, name, text in similar_products:
            print(f"\nID: {unique_id}")
            print(f"Product: {name}")
            print(f"Similarity: {similarity:.3f}")
            print(f"Description: {text[:100]}...")  # Show first 100 chars of description

if __name__ == "__main__":
    main()