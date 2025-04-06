#!/usr/bin/env python3
"""Example Script - Generate Objects Library"""

import argparse
import os
import sys

# Add parent directory to module search path, so it can import prompt_generate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prompt_generator import load_config
from prompt_generator.generators import DeepSeekObjectGenerator  # Directly import the generator class, not the async function

def main() -> int:
    """Main function"""
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Generate Objects Library")
    parser.add_argument("--config", type=str, default="config.yml", help="Configuration file path")
    parser.add_argument("--objects_count", type=int, help="Number of objects to generate per category")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--concurrent_limit", type=int, help="concurrent_limit: Maximum number of concurrent API requests")
    
    args = parser.parse_args()
    
    # If another configuration file is specified, reload
    if args.config != "config.yml":
        config = load_config(args.config)
    
    # Merge command line arguments and configuration file
    objects_count = args.objects_count or config.get('objects')
    output_file = args.output_file or config.get('objects_file', 'objects.jsonl')
    concurrent_limit = args.concurrent_limit
    base_url = config.get('base_url')
    api_key = config.get("api_key")
    model = config.get("model")
    
    # Check API key file
    if not api_key:
        print("Error: API key file not provided")
        return 1
    
    
    try:
        # Synchronous implementation
        print(f"\n====== Generating Objects Library: {output_file} ======")
        
        # Create generator object
        generator = DeepSeekObjectGenerator(
            api_key=api_key,
            default_objects_per_category=objects_count,
            concurrent_limit=concurrent_limit,
            base_url=base_url,
            model=model
        )
        
        # Generate specified number of objects for each category
        custom_counts = {category: objects_count for category in generator.categories}
        
        # Directly call synchronous method, not using asyncio
        generator.generate_and_save_objects_stream(
            output_file=output_file, 
            category_counts=custom_counts
        )
        
        print(f"Objects library generated: {output_file}")
        return 0
    except KeyboardInterrupt:
        print("\nProgram was interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
