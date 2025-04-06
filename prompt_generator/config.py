"""Configuration Loading and Processing Module"""

import os
import yaml
from typing import Dict, Any, Optional

def load_config(config_file: str = 'config.yml') -> Dict[str, Any]:
    """Load YAML configuration file
    
    Args:
        config_file: Configuration file path, defaults to 'config.yml'
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Determine the absolute path of the configuration file
    if not os.path.isabs(config_file):
        # Get the directory of the current module
        module_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(module_dir, config_file)
    else:
        config_path = config_file
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or {}
        else:
            print(f"Configuration file '{config_path}' does not exist!")
            # Try to find in the current working directory
            if config_file != config_path and os.path.exists(config_file):
                print(f"Attempting to load configuration file from current working directory: '{config_file}'")
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config or {}
            return {}
    except Exception as e:
        print(f"Warning: Unable to load configuration file {config_path}: {str(e)}")
        return {}

def get_api_key_from_file(api_key_file: str) -> Optional[str]:
    """Read API key from file
    
    Args:
        api_key_file: API key file path
        
    Returns:
        Optional[str]: API key string or None (if reading fails)
    """
    # If the path is not absolute, try to find from the module directory
    if not os.path.isabs(api_key_file):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(module_dir, api_key_file)
        if os.path.exists(full_path):
            api_key_file = full_path
    
    try:
        with open(api_key_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error: Unable to read API key file {api_key_file}: {str(e)}")
        return None
