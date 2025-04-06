#!/usr/bin/env python3
"""Example Script - Generate Attributes Library"""

import argparse
import os
import sys

# Add parent directory to module search path, so it can import prompt_generate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prompt_generator import load_config
from prompt_generator.generators import DeepSeekAttributeManager  # Directly import the generator class, not the async function

def main() -> int:
    """Main function"""
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Generate Attributes Library")
    parser.add_argument("--config", type=str, default="config.yml", help="Configuration file path")
    parser.add_argument("--attributes_count", type=int, help="Number of attribute concepts to generate for each object")
    parser.add_argument("--concurrent_limit", type=int, help="concurrent_limit: Maximum number of concurrent API requests")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--input_file", type=str, help="Input file path")
    
    args = parser.parse_args()
    
    # If another configuration file is specified, reload
    if args.config != "config.yml":
        config = load_config(args.config)
    
    # Merge command line arguments and configuration file
    attributes_count = args.attributes_count or config.get('attributes', 5)
    input_file = args.input_file or config.get('objects_input')
    output_file = args.output_file or config.get('attributes_file', 'attributes_library.jsonl')
    concurrent_limit = args.concurrent_limit
    base_url = config.get('base_url')
    api_key = config.get("api_key")
    model = config.get("model")
    
    # Check API key file
    if not api_key:
        print("Error: API key not provided")
        return 1
    
    try:
        # Synchronous implementation
        print(f"\n====== Generating Attributes Library: {output_file} ======")
        
        manager = DeepSeekAttributeManager(
        api_key=api_key,
        concurrent_limit=concurrent_limit,
        values_per_attribute=attributes_count,
        base_url=base_url,
        model=model
    )
        manager.process_all(
            input_file=input_file,
            output_file=output_file
    )
        print(f"Attributes library generated: {output_file}")
        
        return 0
    except KeyboardInterrupt:
        print("\nProgram was interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
