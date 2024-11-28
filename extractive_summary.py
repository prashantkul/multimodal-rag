from summarizer import Summarizer
from typing import List, Optional, Tuple, Dict
import re
from tqdm import tqdm
import numpy as np
from transformers import CLIPTokenizer
from product import Product, DataManager
import pandas as pd

class TwoStageBertSummarizer:
    """
    Two-stage BERT summarizer for handling both product names and descriptions.
    """
    
    def __init__(self, clip_tokenizer_name: str = "openai/clip-vit-base-patch32"):
        self.bert_summarizer = Summarizer('distilbert-base-uncased')
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_name)
        
        # Define key patterns
        self.name_patterns = {
            'brand': r'(?:(?:^|\s)(?:by\s)?([A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*))',
            'model': r'(?:[A-Z0-9]+-[A-Z0-9]+|[A-Z0-9]+\d[A-Z0-9]*)',
            'size': r'\b\d+(?:\.\d+)?(?:\s*(?:inch|in|cm|mm|GB|TB|oz|ml|L))?\b'
        }
        
        self.key_specs = {
            'dimensions': r'\b\d+(?:\.\d+)?(?:\s*[xX]\s*\d+(?:\.\d+)?)*(?:\s*(?:mm|cm|m|inch|in|ft))?\b',
            'capacity': r'\b\d+(?:\.\d+)?(?:\s*(?:mAh|GB|TB|MB|W|Hz|MP))\b',
            'connectivity': r'\b(?:wifi|bluetooth|wireless|5G|4G|LTE|USB-C|USB)\b',
            'material': r'\b(?:leather|cotton|polyester|metal|plastic|steel|aluminum)\b'
        }
    
    def _count_clip_tokens(self, text: str) -> int:
        """Count CLIP tokens in text."""
        tokens = self.clip_tokenizer(text, return_tensors="pt")
        return len(tokens.input_ids[0])
    
    def _clean_product_name(self, name: str) -> str:
        """Clean and compress product name."""
        # Remove multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Extract key components
        components = []
        
        # Try to extract brand
        brand_match = re.search(self.name_patterns['brand'], name)
        if brand_match:
            components.append(brand_match.group(1))
        
        # Extract model number
        model_match = re.search(self.name_patterns['model'], name)
        if model_match:
            components.append(model_match.group())
        
        # Extract size/capacity if present
        size_match = re.search(self.name_patterns['size'], name)
        if size_match:
            components.append(size_match.group())
        
        # If no components were extracted, use first 5 words of original name
        if not components:
            components = name.split()[:5]
        
        return ' '.join(components)
    
    def _extract_key_specs(self, text: str) -> Dict[str, List[str]]:
        """Extract key specifications from text."""
        specs = {}
        for spec_type, pattern in self.key_specs.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            specs[spec_type] = list(set(match.group() for match in matches))
        return specs
    
    def summarize_name(self, name: str, max_tokens: int = 20) -> str:
        """Summarize product name to fit within token limit."""
        # If name already fits, return it
        if self._count_clip_tokens(name) <= max_tokens:
            return name
        
        # Clean and compress name
        cleaned_name = self._clean_product_name(name)
        
        # If cleaned name fits, return it
        if self._count_clip_tokens(cleaned_name) <= max_tokens:
            return cleaned_name
        
        # If still too long, truncate while preserving key information
        words = cleaned_name.split()
        for i in range(len(words), 0, -1):
            truncated = ' '.join(words[:i])
            if self._count_clip_tokens(truncated) <= max_tokens:
                return truncated
        
        return words[0]  # Return at least the first word
    
    def summarize_description(self, 
                            description: str, 
                            max_tokens: int = 57) -> str:
        """Summarize product description to fit within token limit."""
        # remove | from description
        description = description.replace('|', '')
        
        # Extract key specifications
        specs = self._extract_key_specs(description)
        
        # Create spec summary
        spec_summary = []
        for spec_type, values in specs.items():
            if values:
                spec_summary.append(f"{spec_type}: {values[0]}")
        
        # Generate BERT summary
        bert_summary = self.bert_summarizer(description, ratio=0.3)
        
        # Combine specs and BERT summary
        combined_text = []
        
        # Add specs first (usually more important)
        spec_text = '; '.join(spec_summary)
        if self._count_clip_tokens(spec_text) <= max_tokens:
            combined_text.append(spec_text)
        
        # Add BERT summary with remaining tokens
        remaining_tokens = max_tokens - (self._count_clip_tokens(' '.join(combined_text)) if combined_text else 0)
        if remaining_tokens > 0:
            words = bert_summary.split()
            for i in range(len(words), 0, -1):
                test_summary = ' '.join(words[:i])
                test_text = ' '.join(combined_text + [test_summary])
                if self._count_clip_tokens(test_text) <= max_tokens:
                    combined_text.append(test_summary)
                    break
        
        return ' '.join(combined_text)
    
    def summarize_product(self, product: Product) -> Tuple[str, str]:
        """
        Generate summaries for both product name and description.
        
        Args:
            product: Product instance
            
        Returns:
            Tuple of (summarized_name, summarized_description)
        """
        # Allocate tokens: 20 for name, 57 for description (total 77)
        name_summary = self.summarize_name(product.name, max_tokens=20)
        desc_summary = self.summarize_description(product.about_product, max_tokens=57)
        
        return name_summary, desc_summary
    
    def batch_process_products(self, products: List[Product]) -> List[Tuple[str, str]]:
        """Process multiple products with progress tracking."""
        summaries = []
        failed_products = []
        
        for product in tqdm(products, desc="Processing products"):
            try:
                summary = self.summarize_product(product)
                summaries.append(summary)
            except Exception as e:
                print(f"Error processing product {product.name}: {str(e)}")
                failed_products.append(product.name)
                summaries.append(("", ""))
        
        # Print statistics
        print(f"\nProcessed {len(products)} products:")
        print(f"Successful: {len(products) - len(failed_products)}")
        print(f"Failed: {len(failed_products)}")
        
        if failed_products:
            print("\nFailed products:")
            for name in failed_products:
                print(f"- {name}")
        
        return summaries

def test_summarizer():
    """Test the summarizer with sample products."""
    # Initialize
    data_manager = DataManager()
    summarizer = TwoStageBertSummarizer()
    
    # Get sample products
    products = data_manager.get_sample_products(5)
    
    print("Processing sample products...\n")
    
    for product in products:
        print(f"Original Product Name ({len(product.name)} chars):")
        print(product.name)
        print(f"\nOriginal Description ({len(product.about_product)} chars):")
        print(product.about_product[:100] + "...\n")
        
        # Generate summaries
        name_summary, desc_summary = summarizer.summarize_product(product)
        
        print("Summarized Name:")
        print(name_summary)
        print(f"Name Tokens: {summarizer._count_clip_tokens(name_summary)}")
        
        print("\nSummarized Description:")
        print(desc_summary)
        print(f"Description Tokens: {summarizer._count_clip_tokens(desc_summary)}")
        
        print(f"\nTotal Tokens: {summarizer._count_clip_tokens(name_summary + ' ' + desc_summary)}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    test_summarizer()