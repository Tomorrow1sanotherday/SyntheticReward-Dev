#!/usr/bin/env python3
"""Example Script - Generate prompts library"""

import argparse
import os
import sys
import asyncio

# Add parent directory to module search path, so it can import prompt_generate
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from prompt_generator import load_config
from prompt_generator.generators import PromptGenerator  # Directly import the generator class, not the async function

def main() -> int:
    """Main function"""
    # Load configuration
    config = load_config()
    
    parser = argparse.ArgumentParser(description="Generate prompts library")
    parser.add_argument("--config", type=str, default="config.yml", help="Configuration file path")
    parser.add_argument("--complexity", type=str, help="Complexity of generated prompts")
    parser.add_argument("--concurrent_limit", type=int, help="concurrent_limit: Maximum number of concurrent API requests")
    parser.add_argument("--output_file", type=str, help="Output file path")
    parser.add_argument("--input_file", type=str, help="Input file path")
    parser.add_argument("--prompts_count", type=int, help="Number of prompts to generate for each object")
    
    args = parser.parse_args()
    
    # If another configuration file is specified, reload
    if args.config != "config.yml":
        config = load_config(args.config)
    
    # Merge command line arguments and configuration file
    complexity = args.complexity or config.get('complexit')
    input_file = args.input_file or config.get('attributes_input')
    output_file = args.output_file or config.get('prompts_file', 'prompts.jsonl')
    prompts_count = args.prompts_count or config.get("prompts")
    concurrent_limit = args.concurrent_limit
    base_url = config.get('base_url')
    api_key = config.get("api_key")
    model = config.get('model')

    
    # Check API key file
    if not api_key:
        print("Error: API key not provided")
        return 1
    
    
    try:
        prompt_generator = PromptGenerator(
            jsonl_file_path=input_file,
            api_key=api_key,
            concurrent_limit=concurrent_limit,
            base_url=base_url,
            model=model
    )
    
        asyncio.run(prompt_generator.generate_all_prompts(
                complexity=complexity,  # Complexity range 0 to 3 (inclusive left, exclusive right)
                prompts_per_object=prompts_count,  # Generate 2 prompts per object
                output_file=output_file
            ))
        print(f"Prompts generated: {output_file}")
        return 0
    except KeyboardInterrupt:
        print("\nProgram was interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
