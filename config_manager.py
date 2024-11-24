import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigManager:
    """
    Manages configuration loading and validation for the image processing pipeline.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the ConfigManager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        config_path = Path("config" + os.sep + config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        print("Config path:", config_path)
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration values
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing config file: {str(e)}")
    
    def _validate_config(self):
        """Validate required configuration values."""
        # Required fields in config
        required_fields = {
            'gcp': ['project_id', 'bucket_name'],
            'image_processing': ['max_workers', 'max_retries', 'timeout'],
            'amazon_products': ['csv_path', 'storage', 'gcs_prefix']
        }
        
        # Check for required fields
        for section, fields in required_fields.items():
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
            
            for field in fields:
                if field not in self.config[section]:
                    raise ValueError(f"Missing required field: {section}.{field}")
        
        # Validate specific values
        if self.config['image_processing']['max_workers'] < 1:
            raise ValueError("max_workers must be at least 1")
        if self.config['image_processing']['max_retries'] < 0:
            raise ValueError("max_retries must be non-negative")
        if self.config['image_processing']['timeout'] < 1:
            raise ValueError("timeout must be at least 1 second")
    
    def get_gcp_config(self) -> Dict[str, str]:
        """Get GCP-related configuration."""
        return self.config['gcp']
    
    def get_image_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration."""
        return self.config['image_processing']
    
    def get_amazon_config(self) -> Dict[str, Any]:
        """Get Amazon products configuration."""
        return self.config['amazon_products']
    
    @property
    def project_id(self) -> str:
        """Get GCP project ID."""
        return self.config['gcp']['project_id']
    
    @property
    def bucket_name(self) -> str:
        """Get GCS bucket name."""
        return self.config['gcp']['bucket_name']
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration values and save to file.
        
        Args:
            updates: Dictionary containing configuration updates
        """
        # Update configuration
        for section, values in updates.items():
            if section in self.config:
                self.config[section].update(values)
            else:
                self.config[section] = values
        
        # Validate new configuration
        self._validate_config()

# # test
config_manager = ConfigManager()
print(config_manager.get_gcp_config())
print(config_manager.get_image_processing_config())
print(config_manager.get_amazon_config())
print(config_manager.project_id)
print(config_manager.bucket_name)
print(config_manager.get_amazon_config()['csv_path'])
print(config_manager.get_amazon_config()['storage'])
print(config_manager.get_amazon_config()['gcs_prefix'])
        
        