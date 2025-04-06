import json
import time
import os
import asyncio
import aiohttp
from typing import List, Dict, Optional, Any, Union, Tuple, AsyncIterator

from tqdm import tqdm
import sys
import random

from tqdm.asyncio import tqdm_asyncio # Use async-compatible tqdm
import traceback # Import traceback for better error logging





class DeepSeekObjectGenerator:
    def __init__(self, api_key: str, base_url: str, model: str,default_objects_per_category: int = 42, concurrent_limit: int = 5):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.default_objects_per_category = default_objects_per_category
        self.concurrent_limit = concurrent_limit
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        self.categories = [
            "Natural Landscapes",
            "Cities and Architecture",
            "People",
            "Animals",
            "Plants",
            "Food and Beverages",
            "Sports and Fitness",
            "Art and Culture",
            "Technology and Industry",
            "Everyday Objects",
            "Transportation",
            "Abstract and Conceptual Art"
        ]
    
    def _read_api_key(self, api_key_file: str) -> str:
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Unable to read API key from file {api_key_file}: {str(e)}")
    
    async def generate_objects_for_category_stream(self, session: aiohttp.ClientSession, category: str, count: int, max_retries: int = 7) -> AsyncIterator[Tuple[str, str]]:
        prompt_instruction = f"""# Object Generation Task
Generate {count} different object names for the "{category}" category.
## Requirements:
- Each object should be represented by a single word or short phrase (e.g., "dog", "cat", "mountain")
- Object names should be diverse and cover different aspects of the category
- Ensure no duplicates
- Object names should not contain any unnecessary sequences like numbers or "object_" tags
- Return only object names, one per line, without any additional description
        """
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise object name generator."},
                {"role": "user", "content": prompt_instruction}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": True
        }
        
        retries = 0
        while retries <= max_retries:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    
                    buffer = ""
                    object_count = 0
                    
                    async for chunk in response.content:
                        if object_count >= count:
                            break
                            
                        try:
                            chunk_text = chunk.decode('utf-8')
                            
                            if 'data: ' in chunk_text:
                                json_parts = [part.strip() for part in chunk_text.split('data: ') if part.strip()]
                                
                                for json_part in json_parts:
                                    if json_part == '[DONE]':
                                        continue
                                    
                                    try:
                                        data = json.loads(json_part)
                                        if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                                            content = data['choices'][0]['delta'].get('content', '')
                                            if content:
                                                buffer += content
                                                
                                                lines = buffer.split('\n')
                                                if len(lines) > 1:
                                                    for line in lines[:-1]:
                                                        cleaned_line = line.strip()
                                                        if cleaned_line and object_count < count:
                                                            object_count += 1
                                                            yield category, cleaned_line
                                                    
                                                    buffer = lines[-1]
                                    except json.JSONDecodeError:
                                        continue
                        except UnicodeDecodeError:
                            continue
                    
                    if buffer.strip() and object_count < count:
                        yield category, buffer.strip()
                        object_count += 1
                    
                    if object_count < count:
                        retries += 1
                        wait_time = 0
                        print(f"Warning: Generated fewer objects than requested for {category}. Retrying... {retries} times")
                        await asyncio.sleep(wait_time)
                    else:
                        return
                        
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    wait_time = 0
                    print(f"Error generating objects via stream for {category}: {str(e)}, retry {retries} (waiting {wait_time} seconds)")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error generating objects via stream for {category}: {str(e)}, max retries reached, giving up")
                    return
    
    async def process_category_stream(self, output_file: str, category: str, count: int, semaphore: asyncio.Semaphore):
        async with semaphore:
            print(f"Generating {count} objects for {category}...")
            cat_id = self.categories.index(category) + 1
            
            category_json = {
                "category_id": cat_id,
                "category_name": category,
                "objects": []
            }
            
            obj_id = 1
            async with aiohttp.ClientSession() as session:
                async for _, obj_name in self.generate_objects_for_category_stream(session, category, count):
                    category_json["objects"].append({
                        "id": obj_id,
                        "name": obj_name
                    })
                    obj_id += 1
                    
                    print(f"Category '{category}' generated: {obj_id-1}/{count} objects")
            
            async with self.file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(category_json, ensure_ascii=False) + '\n')
                print(f"Wrote {len(category_json['objects'])} objects for category '{category}' to file")
    
    async def generate_and_save_objects_stream_async(self, output_file: str = "category_objects_stream.jsonl", category_counts: Optional[Dict[str, int]] = None) -> None:
        start_time = time.time()
        print(f"Starting stream object generation and writing to {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        if category_counts is None:
            category_counts = {category: self.default_objects_per_category for category in self.categories}
        
        for category in self.categories:
            if category not in category_counts:
                category_counts[category] = self.default_objects_per_category
        
        self.file_lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        
        tasks = [
            self.process_category_stream(output_file, category, category_counts[category], semaphore)
            for category in self.categories
        ]
        
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        print(f"Objects for all categories have been stream generated and written to {output_file}")
        print(f"Total time: {elapsed_time:.2f} seconds")
    
    def generate_and_save_objects_stream(self, output_file: str = "category_objects_stream.jsonl", category_counts: Optional[Dict[str, int]] = None) -> None:
        asyncio.run(self.generate_and_save_objects_stream_async(output_file, category_counts))



class DeepSeekAttributeManager:
    def __init__(self, api_key: str, model: str, base_url: str = "https://api.nuwaapi.com/v1", concurrent_limit: int = 3, values_per_attribute: int = 8):
        """
        Initialize DeepSeek Attribute Manager

        Parameters:
            api_key: The API key string.
            model: The model name to use.
            base_url: Base URL for DeepSeek API
            concurrent_limit: Maximum number of concurrent API requests
            values_per_attribute: Number of values to generate per attribute (5-10)
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # Set a minimum concurrent limit to avoid potential issues with very low values
        self.concurrent_limit = max(1, concurrent_limit)
        # Ensure values_per_attribute is within the desired range, adjust if necessary
        self.values_per_attribute = max(5, min(10, values_per_attribute))
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Add file lock for thread-safe concurrent file writing from different category tasks
        self.file_lock = asyncio.Lock()

    # Note: _read_api_key is not needed if api_key is passed directly
    # def _read_api_key(self, api_key_file: str) -> str:
    #     """Read API key from text file"""
    #     try:
    #         with open(api_key_file, 'r', encoding='utf-8') as f:
    #             return f.read().strip()
    #     except Exception as e:
    #         raise ValueError(f"Cannot read API key from file {api_key_file}: {str(e)}")

    def read_jsonl(self, input_file: str) -> List[Dict[str, Any]]:
        """Read JSONL file, return list of parsed data"""
        data = []
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                             print(f"Warning: Skipping invalid JSON line in {input_file}: {e} - Line: '{line.strip()}'")
            return data
        except FileNotFoundError:
             print(f"Error: Input file not found: {input_file}")
             return []
        except Exception as e:
            print(f"Error when reading JSONL file {input_file}: {str(e)}")
            return []

    async def write_jsonl_line(self, output_file: str, data: Dict[str, Any]) -> None:
        """Safely write a single line of data to JSONL file using an async lock"""
        async with self.file_lock:
            try:
                # Use 'a' mode to append lines
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written to disk immediately
                    os.fsync(f.fileno()) # Force write to disk, more robust
            except Exception as e:
                print(f"Error when writing JSONL data to {output_file}: {str(e)}")

    async def _call_streaming_api_and_accumulate(self, session: aiohttp.ClientSession, url: str, payload: Dict[str, Any], timeout: aiohttp.ClientTimeout) -> str:
        """
        Helper method to call the streaming API and accumulate the full response content.
        Handles chunk processing and potential JSON structure within stream data.
        """
        full_response_content = ""
        async with session.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=timeout
        ) as response:
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            buffer = "" # Buffer for handling partial JSON objects across chunks
            async for chunk in response.content:
                try:
                    chunk_text = chunk.decode('utf-8')
                    # Process potential server-sent event format
                    if 'data: ' in chunk_text:
                        json_parts = [part.strip() for part in chunk_text.split('data: ') if part.strip()]
                        for json_part in json_parts:
                            if json_part == '[DONE]':
                                continue
                            try:
                                # Prepend buffer in case the previous chunk ended mid-object
                                complete_json_str = buffer + json_part
                                data = json.loads(complete_json_str)
                                buffer = "" # Reset buffer after successful parse
                                if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                                    content = data['choices'][0]['delta'].get('content', '')
                                    if content:
                                        full_response_content += content
                            except json.JSONDecodeError:
                                # If decode fails, it might be a partial object, store in buffer
                                buffer += json_part
                                continue # Skip to next chunk/part
                    else:
                         # Handle plain text chunks if API sends them differently (less common for chat completions)
                         # For simplicity, we assume 'data: ' format, but could add logic here if needed
                         pass # Or accumulate directly: full_response_content += chunk_text

                except UnicodeDecodeError:
                    print("Warning: UnicodeDecodeError processing chunk, skipping.")
                    continue

            # Attempt to parse any remaining buffer content (though ideally it should be empty)
            if buffer.strip():
                 print(f"Warning: Stream ended with non-empty buffer: {buffer}")
                 # Optionally try to parse it or append if it looks like plain text
                 try:
                    # Attempt final parse of buffer content if it seems like a JSON delta
                    data = json.loads(buffer)
                    if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                        content = data['choices'][0]['delta'].get('content', '')
                        if content:
                            full_response_content += content
                 except json.JSONDecodeError:
                     # If it's not JSON, maybe append directly or log warning
                     # full_response_content += buffer # Decide if needed
                     pass

        return full_response_content

    async def determine_object_attributes(self,
                                        session: aiohttp.ClientSession,
                                        category: str,
                                        object_name: str) -> List[str]:
        """Query DeepSeek API (streaming) to determine appropriate attributes for a specific object. Retries on failure."""
        prompt_instruction = f"""# Attribute Identification Task
I need to determine appropriate attributes for the object **"{object_name}"** in the category **"{category}"**.
Please list 10-12 attribute names that best describe this object, based on reliable knowledge sources (such as Wikipedia).
## Guidelines
Ensure the attributes are appropriate for the object. For example:
- For "dog", suitable attributes might be: breed, color, size, temperament, sound, etc.
- For "mountain range", suitable attributes might be: height, geological composition, climate, vegetation cover, geographical location, etc.
- For "building", suitable attributes might be: architectural style, materials, height, era, function, etc.
## Output Format
Return the attribute names as a JSON array, with each attribute expressed as an English word:
["attribute1", "attribute2", "attribute3", ...]
Do not add any explanatory text.
Examples:[breed, color, size, ...]"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional object attribute analyzer. Please return only a JSON list of attribute names without any explanations."},
                {"role": "user", "content": prompt_instruction}
            ],
            "temperature": 0.7,
            "max_tokens": 300,
            # "response_format": {"type": "json_object"} # Remove for streaming
            "stream": True # Enable streaming
        }

        wait_time = 0 # Wait time between retries (adjust if needed)
        max_retries = 10 # Limit retries to avoid infinite loops in persistent failure scenarios
        retries = 0
        api_url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=60) # Timeout for the request

        while retries < max_retries:
            try:
                # Call the streaming helper method
                attributes_text = await self._call_streaming_api_and_accumulate(
                    session, api_url, payload, timeout
                )

                # Check if the accumulated text is empty
                if not attributes_text.strip():
                     print(f"Warning: API returned empty content stream for attributes of {object_name} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                     retries += 1
                     await asyncio.sleep(wait_time)
                     continue # Retry

                # print(f"Accumulated response (attributes for {object_name}): {attributes_text}") # Verbose logging

                # Parse the accumulated JSON text
                try:
                    attributes_data = json.loads(attributes_text)
                    attributes_list = []

                    # Try to extract the list from various possible structures (existing logic)
                    if isinstance(attributes_data, list):
                        attributes_list = attributes_data
                    elif isinstance(attributes_data, dict):
                        found = False
                        for key in ["attributes", "attribute_names", "result", "data"]:
                            if key in attributes_data and isinstance(attributes_data[key], list):
                                attributes_list = attributes_data[key]
                                found = True
                                break
                        if not found:
                            for value in attributes_data.values():
                                 if isinstance(value, list):
                                     attributes_list = value
                                     found = True
                                     break
                        if not found:
                            print(f"Warning: Cannot extract attribute list from parsed JSON dict for {object_name}: {attributes_data} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                            retries += 1
                            await asyncio.sleep(wait_time)
                            continue # Retry
                    else:
                         print(f"Warning: Unexpected format after parsing JSON for attributes of {object_name}: {type(attributes_data)} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                         retries += 1
                         await asyncio.sleep(wait_time)
                         continue # Retry

                    # Check if the list is valid (not empty and has enough items)
                    if not attributes_list or len(attributes_list) < 3:
                        print(f"Warning: Attribute list from parsed JSON for {object_name} is empty or too short ({len(attributes_list)}) (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                        retries += 1
                        await asyncio.sleep(wait_time)
                        continue # Retry

                    # Successfully got a valid list
                    print(f"Successfully determined attributes for {object_name}") #: {attributes_list}")
                    return [str(attr).strip() for attr in attributes_list if str(attr).strip()] # Clean and return

                except json.JSONDecodeError:
                    print(f"Warning: Cannot parse accumulated JSON from stream for {object_name}: '{attributes_text}' (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                    retries += 1
                    await asyncio.sleep(wait_time)
                    continue # Retry

            except aiohttp.ClientResponseError as e:
                 print(f"HTTP Error determining attributes for {object_name}: {e.status} {e.message} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                 retries += 1
                 await asyncio.sleep(wait_time)
                 continue # Retry
            except asyncio.TimeoutError:
                 print(f"Timeout error determining attributes for {object_name} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                 retries += 1
                 await asyncio.sleep(wait_time)
                 continue # Retry
            except Exception as e:
                # Catch other potential errors (network issues, etc.)
                print(f"Unexpected error determining attributes for {object_name}: {str(e)} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                retries += 1
                await asyncio.sleep(wait_time)
                continue # Retry

        # If loop finishes without returning, max retries were reached
        print(f"!!!!!!!! FAILED to determine attributes for {object_name} after {max_retries} retries. !!!!!!!!")
        return [] # Return empty list indicates failure

    async def generate_attribute_values(self,
                                     session: aiohttp.ClientSession,
                                     category: str,
                                     object_name: str,
                                     attribute_name: str,
                                     count: int) -> List[str]:
        """Use DeepSeek API (streaming) to generate multiple possible values for a specific attribute. Retries on failure."""
        prompt_instruction = f"""# Attribute Value Generation Task
I need to generate {count} different possible values for the attribute **"{attribute_name}"** of object **"{object_name}"** in category **"{category}"**.
Please provide accurate and diverse values based on reliable knowledge sources (such as Wikipedia). Ensure these values:
1. Are factually accurate and consistent with objective reality
2. Are concise and clear, with each value not exceeding 5 words
3. Are specific and detailed, avoiding overly general descriptions
4. Differ from each other, covering various possibilities
## Examples
- For the "height" attribute of "mountain", possible values include: "over 8000 meters", "medium-sized hills", "towering peaks", "sea-level mountains", "dramatic cliffs", "gentle slopes", "alpine summits", "imposing heights"
- For the "type" attribute of "tree", possible values include: "oak", "pine", "maple", "willow", "redwood", "bamboo", "birch", "palm"
## Output Format
Return as a JSON array in this format: ["value1", "value2", "value3", ...]
Do not add any explanatory text.
Examples:["oak", "pine", ...]"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a professional attribute value generation expert. You will use your knowledge base to generate accurate attribute values. Output only a JSON list of attribute values without any explanation."},
                {"role": "user", "content": prompt_instruction}
            ],
            "temperature": 0.7,
            "max_tokens": 400,
            # "response_format": {"type": "json_object"} # Remove for streaming
            "stream": True # Enable streaming
        }

        wait_time = 0 # Wait time between retries (adjust if needed)
        max_retries = 10 # Limit retries
        retries = 0
        api_url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=60) # Timeout for the request

        while retries < max_retries:
            try:
                 # Call the streaming helper method
                values_text = await self._call_streaming_api_and_accumulate(
                    session, api_url, payload, timeout
                )

                # Check if the accumulated text is empty
                if not values_text.strip():
                     print(f"Warning: API returned empty content stream for values of {attribute_name} of {object_name} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                     retries += 1
                     await asyncio.sleep(wait_time)
                     continue # Retry

                # print(f"Accumulated response (values for {object_name} -> {attribute_name}): {values_text}") # Verbose logging

                # Parse the accumulated JSON text
                try:
                    values_data = json.loads(values_text)
                    values_list = []

                    # Try to extract the list from various possible structures (existing logic)
                    if isinstance(values_data, list):
                        values_list = values_data
                    elif isinstance(values_data, dict):
                        found = False
                        for key in ["values", "value", "attributes", "options", "result", "data"]:
                             if key in values_data and isinstance(values_data[key], list):
                                 values_list = values_data[key]
                                 found = True
                                 break
                        if not found:
                             for value in values_data.values():
                                 if isinstance(value, list):
                                     values_list = value
                                     found = True
                                     break
                        if not found:
                            print(f"Warning: Unable to extract value list from parsed JSON dict for {attribute_name} of {object_name}: {values_data} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                            retries += 1
                            await asyncio.sleep(wait_time)
                            continue # Retry
                    else:
                        print(f"Warning: Unexpected format after parsing JSON for values of {attribute_name} of {object_name}: {type(values_data)} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                        retries += 1
                        await asyncio.sleep(wait_time)
                        continue # Retry


                    # Check if the list is valid (not empty)
                    if not values_list:
                        print(f"Warning: API returned empty list (after parsing) for {attribute_name} of {object_name} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                        # Decide if empty list is acceptable or requires retry. Let's retry.
                        retries += 1
                        await asyncio.sleep(wait_time)
                        continue # Retry

                    # Check if we got enough values, but proceed even if fewer were returned
                    if len(values_list) < count:
                        print(f"Warning: API returned only {len(values_list)} values for {attribute_name} of {object_name} (expected {count}). Using available values.")
                        # Don't retry automatically, accept the values received unless list is completely empty (handled above)

                    # Successfully got a valid list
                    # Convert all values to string, strip whitespace, filter out empty strings, take only the required count
                    final_values = [str(value).strip() for value in values_list if str(value).strip()][:count]

                    if not final_values:
                         print(f"Warning: After cleaning, no valid values remained for {attribute_name} of {object_name} (Retry {retries+1}/{max_retries}), retrying...")
                         retries += 1
                         await asyncio.sleep(wait_time)
                         continue # Retry

                    print(f"Successfully generated {len(final_values)} values for {object_name} -> {attribute_name}") #: {final_values}")
                    return final_values # Exit loop and return result

                except json.JSONDecodeError:
                    print(f"Warning: Cannot parse accumulated JSON from stream for {attribute_name} of {object_name}: '{values_text}' (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                    retries += 1
                    await asyncio.sleep(wait_time)
                    continue # Retry

            except aiohttp.ClientResponseError as e:
                 print(f"HTTP Error generating values for {attribute_name} of {object_name}: {e.status} {e.message} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                 retries += 1
                 await asyncio.sleep(wait_time)
                 continue # Retry
            except asyncio.TimeoutError:
                 print(f"Timeout error generating values for {attribute_name} of {object_name} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                 retries += 1
                 await asyncio.sleep(wait_time)
                 continue # Retry
            except Exception as e:
                # Catch other potential errors (network issues, etc.)
                print(f"Unexpected error generating values for {attribute_name} of {object_name}: {str(e)} (Retry {retries+1}/{max_retries}), retrying in {wait_time} seconds...")
                retries += 1
                await asyncio.sleep(wait_time)
                continue # Retry

        # If loop finishes without returning, max retries were reached
        print(f"!!!!!!!! FAILED to generate values for {attribute_name} of {object_name} after {max_retries} retries. !!!!!!!!")
        return [] # Return empty list indicates failure

    async def process_object(self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore,
                         category_name: str, obj_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Complete processing of a single object: determine attribute list and generate values for each attribute.
        Returns the updated object data or None if processing fails at a critical step.
        """
        object_name = obj_data.get("name", "Unnamed Object") # Use .get for safety
        print(f"Starting processing: {category_name} -> {object_name}")

        try:
            # 1. Determine appropriate attributes for this object
            # This part needs the semaphore as it makes an API call
            attribute_list: List[str] = []
            async with semaphore:
                print(f"Determining attributes for: {category_name} -> {object_name}")
                # Calls the modified determine_object_attributes which now uses streaming
                attribute_list = await self.determine_object_attributes(
                    session, category_name, object_name
                )
                # Check if determining attributes failed (returned empty list)
                if not attribute_list:
                    print(f"!!!!!!!! FAILED critical step: could not determine attributes for object {object_name} in category {category_name}. Skipping this object. !!!!!!!!")
                    return None # Indicate failure

            # 2. Generate values for each attribute
            updated_attributes = {}
            value_tasks = []
            if attribute_list: # Proceed only if we have attributes
                for attr in attribute_list:
                    # Define a coroutine function for generating values for a single attribute
                    # This function will acquire the semaphore internally for its API call
                    async def generate_values_for_attr(attribute_name):
                        async with semaphore: # Acquire semaphore for each value generation API call
                            # print(f"Generating values: {category_name} -> {object_name} -> {attribute_name}") # Less verbose log
                            # Calls the modified generate_attribute_values which now uses streaming
                            values = await self.generate_attribute_values(
                                session, category_name, object_name, attribute_name, self.values_per_attribute
                            )
                            # Return even if values list is empty (failure logged inside generate_attribute_values)
                            return attribute_name, values

                    value_tasks.append(generate_values_for_attr(attr))

                # Run value generation tasks concurrently (respecting semaphore limit)
                # Use return_exceptions=True to prevent one failure from stopping others
                results = await asyncio.gather(*value_tasks, return_exceptions=True)

                # Collect results, handling potential exceptions during value generation
                successful_attributes_count = 0
                for result in results:
                    if isinstance(result, Exception):
                        # Log the exception if needed, error primarily logged in gather target
                        print(f"!!!!!!!! Error occurred during value generation for an attribute of {object_name}: {result}. Skipping this attribute's result. !!!!!!!!")
                    elif isinstance(result, tuple) and len(result) == 2:
                        attr_name, values = result
                        if values: # Check if values list is not empty (indicates success)
                            updated_attributes[attr_name] = values
                            successful_attributes_count += 1
                        else:
                             # Failure to get values for this attribute was already logged inside generate_attribute_values
                             print(f"Info: No values were successfully generated for attribute '{attr_name}' of {object_name}.")
                    else:
                        print(f"Warning: Unexpected result type from generate_values_for_attr for {object_name}: {type(result)}")


                # Check if we got at least some attributes successfully
                if not updated_attributes:
                     print(f"!!!!!!!! FAILED: No attribute values could be successfully generated for object {object_name} in category {category_name}. Skipping this object. !!!!!!!!")
                     return None # Indicate failure

                print(f"✓ Object processing completed: {category_name} -> {object_name} ({successful_attributes_count}/{len(attribute_list)} attributes processed)")

            else: # Should not happen due to check after determine_object_attributes, but as safety
                print(f"Warning: No attributes determined for {object_name}, cannot generate values.")
                # Decide if this is a failure case. If attributes are essential, return None.
                # Let's consider it a failure if the initial step yielded nothing.
                return None


            # 3. Update object data
            updated_obj = obj_data.copy()
            updated_obj["attributes"] = updated_attributes
            return updated_obj

        except Exception as e:
             # Catch exceptions from semaphore acquisition or other unexpected errors in this scope
             print(f"!!!!!!!! UNEXPECTED FAILED during processing of object {object_name} in category {category_name}: {e}. Skipping this object. !!!!!!!!")
             traceback.print_exc() # Log the full traceback
             return None # Indicate failure

    async def process_category_stream(self, category_data: Dict[str, Any], semaphore: asyncio.Semaphore,
                                 output_file: str) -> None:
        """Stream process all objects in a single category concurrently and write the result once."""
        category_name = category_data.get("category_name", "Unnamed Category")
        objects = category_data.get("objects", [])

        if not objects:
             print(f"Skipping category '{category_name}': No objects found.")
             return

        # Ensure category_id exists
        if "category_id" not in category_data:
            category_data["category_id"] = f"cat_{category_name.lower().replace(' ', '_')}"

        print(f"--- Starting concurrent processing for category: '{category_name}' ({len(objects)} objects) ---")

        # Prepare the final structure for JSONL output - start with category info
        final_category_output = {k: v for k, v in category_data.items() if k != "objects"}
        final_category_output["objects"] = [] # Initialize with empty list for processed objects

        # Use a single session for all requests within this category processing task
        async with aiohttp.ClientSession() as session:
            # Create tasks for processing each object concurrently
            object_tasks = []
            for obj_data in objects:
                 if not isinstance(obj_data, dict) or "name" not in obj_data:
                      print(f"Warning: Skipping invalid object data in category '{category_name}': {obj_data}")
                      continue
                 # Pass the semaphore and session to each object processing task
                 task = asyncio.create_task(
                     self.process_object(session, semaphore, category_name, obj_data),
                     name=f"ProcessObject_{category_name}_{obj_data.get('name','Unknown')}" # Name task for debugging
                 )
                 object_tasks.append(task)

            if not object_tasks:
                print(f"No valid objects to process in category '{category_name}'.")
                return

            # Run object processing tasks concurrently
            # return_exceptions=True allows us to collect results even if some objects fail
            processed_results = await asyncio.gather(*object_tasks, return_exceptions=True)

        # Collect successfully processed objects and handle errors
        processed_objects_count = 0
        failed_objects_count = 0
        for i, result in enumerate(processed_results):
            task_name = object_tasks[i].get_name() # Get task name for context
            if isinstance(result, Exception):
                # The error should have been logged within process_object or its callees
                print(f"Error captured by gather for task {task_name}: {result}")
                failed_objects_count += 1
            elif result is not None: # process_object returns None on failure
                final_category_output["objects"].append(result)
                processed_objects_count += 1
            else: # result is None, failure logged in process_object
                # print(f"Info: Task {task_name} failed, object skipped (logged previously).")
                failed_objects_count += 1

        # Write the final category data to main output file only if any objects were successfully processed
        if final_category_output["objects"]:
            await self.write_jsonl_line(output_file, final_category_output)
            print(f"✓ Category '{category_name}' processing completed. Saved {processed_objects_count} successful objects (out of {len(objects)} total).")
            if failed_objects_count > 0:
                 print(f"  ({failed_objects_count} objects failed or were skipped)")
        else:
            print(f"✗ Category '{category_name}' processing completed but NO objects were successfully processed or saved ({failed_objects_count}/{len(objects)} failed/skipped).")

    async def process_all_stream(self, input_file: str, output_file: str) -> None:
        """Stream process all categories and objects from JSONL file, writing results concurrently."""
        start_time = time.time()
        print(f"Starting processing...")
        print(f"Reading data from: {input_file}")
        data = self.read_jsonl(input_file)

        if not data:
            print("No valid data found in input file, exiting.")
            return

        print(f"Read {len(data)} categories from input file.")
        total_objects_in_input = sum(len(cat.get("objects", [])) for cat in data)
        print(f"Total objects across all categories in input: {total_objects_in_input}")


        # Create or clear output file before starting tasks
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                pass # Just clear the file if it exists, or create it
            print(f"Cleared/Created output file: {output_file}")
        except Exception as e:
             print(f"FATAL: Error clearing/creating output file {output_file}: {e}. Exiting.")
             return


        # Create semaphore to control overall concurrency for API calls across all tasks
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        print(f"Concurrency limit set to: {self.concurrent_limit} (max simultaneous API calls)")

        # Create tasks for processing each category concurrently
        category_tasks = []
        for i, category in enumerate(data):
             if not isinstance(category, dict) or "category_name" not in category:
                 print(f"Warning: Skipping invalid category data at index {i}: {category}")
                 continue
             print(f"Queueing processing for category {i+1}/{len(data)}: {category.get('category_name', 'Unnamed Category')}")
             # Create a task for each category processing stream
             task = asyncio.create_task(
                 self.process_category_stream(category, semaphore, output_file),
                 name=f"ProcessCategory_{category.get('category_name','Unnamed')}" # Name task
             )
             category_tasks.append(task)

        # Run category processing tasks concurrently
        # The semaphore passed into each task limits the *total* number of concurrent API calls
        # across all running category/object tasks.
        if category_tasks:
            await asyncio.gather(*category_tasks) # No need for return_exceptions here, errors handled within tasks
        else:
             print("No valid categories found to process.")


        print("\n--- Processing complete ---")
        elapsed_time = time.time() - start_time

        # Recalculate total objects successfully processed by reading the output file
        final_processed_objects_count = 0
        final_processed_categories_count = 0
        try:
             with open(output_file, 'r', encoding='utf-8') as f:
                 for line in f:
                     try:
                         if line.strip():
                             category_data = json.loads(line)
                             final_processed_objects_count += len(category_data.get("objects", []))
                             final_processed_categories_count += 1
                     except json.JSONDecodeError:
                         print(f"Warning: Could not parse line in output file during final count: '{line.strip()}'")
        except FileNotFoundError:
             print("Warning: Output file not found after processing for final count.")
        except Exception as e:
            print(f"Warning: Error reading output file for final count: {e}")


        print(f"Processed {final_processed_categories_count} categories.")
        print(f"Successfully processed and saved data for {final_processed_objects_count} objects (out of {total_objects_in_input} in input).")
        print(f"Results saved to: {output_file}")
        print(f"Total time: {elapsed_time:.2f} seconds")

    def process_all(self, input_file: str, output_file: str) -> None:
        """
        Synchronous entry point to process all categories and objects in a JSONL file.
        Handles starting and running the asyncio event loop.
        """
        print("Starting synchronous process_all...")
        # Ensure asyncio event loop runs correctly, especially in environments like Jupyter
        try:
            # Check if a loop is already running (e.g., in Jupyter)
            loop = asyncio.get_running_loop()
            print("Asyncio loop already running. Adding task to the loop.")
            # If a loop exists, create a task and wait for it.
            # This is generally safer than calling asyncio.run inside an existing loop.
            async def main_task():
                await self.process_all_stream(input_file, output_file)
            # Use a more robust way to run if loop is already running
            # loop.run_until_complete(main_task()) # This might block if called from within a running task

            # Best practice: If called from sync context where loop *might* be running,
            # it's often better to just use asyncio.run() and let it manage the loop.
            # If issues arise specifically in Jupyter/IPython, consider `nest_asyncio`.
            # For general script usage, asyncio.run() is preferred.
            asyncio.run(self.process_all_stream(input_file, output_file))

        except RuntimeError as e:
             if "cannot call run_until_complete() when the event loop is running" in str(e):
                 print("Detected running loop issue. Consider using await if in async context, or nest_asyncio in Jupyter.")
                 # As a fallback attempt, try creating task - might not work depending on context
                 # loop = asyncio.get_event_loop()
                 # loop.create_task(self.process_all_stream(input_file, output_file))
                 print("Cannot automatically run in this context. Please call 'await manager.process_all_stream(...)' if in an async function.")

             elif "no running event loop" in str(e):
                 print("No asyncio loop running, starting new one with asyncio.run().")
                 asyncio.run(self.process_all_stream(input_file, output_file))
             else:
                 print(f"An unexpected runtime error occurred: {e}")
                 # Re-raise if it's not a loop-related issue we can handle
                 raise




class PromptGenerator:
    def __init__(self, jsonl_file_path: str, api_key: str, model: str, concurrent_limit: int = 5, base_url: str = "https://api.nuwaapi.com/v1"):
        """
        Initialize the prompt generator

        Parameters:
            jsonl_file_path: Path to JSONL file containing objects and attribute values
            api_key: The API key string.
            model: The model name to use.
            concurrent_limit: Maximum number of concurrent API requests.
            base_url: DeepSeek API base URL.
        """
        self.objects_data = self._load_jsonl(jsonl_file_path)
        if not self.objects_data:
             raise ValueError(f"No data loaded from {jsonl_file_path}. Please check the file.")

        self.categories = list(self.objects_data.keys())

        # Create a category ID mapping
        self.category_id_map = {category: idx + 1 for idx, category in enumerate(self.categories)}

        # Initialize ID counter - will be managed carefully during concurrent generation
        self.total_prompt_id_counter = 1 # Start from 1

        # DeepSeek API settings
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Ensure concurrent limit is at least 1
        self.concurrent_limit = max(1, concurrent_limit)
        # Lock for thread-safe file writing and ID incrementing
        self.file_lock = asyncio.Lock()

    # _read_api_key is removed as api_key is passed directly

    def _load_jsonl(self, jsonl_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load data from JSONL file and organize by category"""
        category_objects: Dict[str, List[Dict[str, Any]]] = {}
        total_objects = 0
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            category_name = data.get("category_name", None)
                            if not category_name:
                                 print(f"Warning: Skipping line {line_num} in {jsonl_file_path} due to missing 'category_name'.")
                                 continue

                            if category_name not in category_objects:
                                category_objects[category_name] = []

                            objects = data.get("objects", [])
                            if not isinstance(objects, list):
                                print(f"Warning: 'objects' field in line {line_num} (category '{category_name}') is not a list. Skipping line.")
                                continue

                            # Basic validation for objects
                            valid_objects = []
                            for obj_idx, obj in enumerate(objects):
                                if isinstance(obj, dict) and "name" in obj and "attributes" in obj:
                                    # Assign a temporary unique ID within the category for processing
                                    obj["temp_cat_obj_id"] = len(category_objects[category_name]) + len(valid_objects) + 1
                                    valid_objects.append(obj)
                                    total_objects += 1
                                else:
                                     print(f"Warning: Skipping invalid object at index {obj_idx} in line {line_num} (category '{category_name}'). Missing 'name' or 'attributes'. Object data: {obj}")

                            category_objects[category_name].extend(valid_objects)

                        except json.JSONDecodeError as e:
                            print(f"Warning: Skipping invalid JSON line {line_num} in {jsonl_file_path}: {e}")
                        except Exception as e:
                             print(f"Warning: Unexpected error processing line {line_num} in {jsonl_file_path}: {e}")

            print(f"Loaded data for {total_objects} objects across {len(category_objects)} categories from '{jsonl_file_path}'")
            if total_objects == 0:
                 print("Warning: No valid objects with names and attributes found in the input file.")
            return category_objects

        except FileNotFoundError:
            print(f"Error: Input file not found: {jsonl_file_path}")
            return {}
        except Exception as e:
            print(f"Error loading JSONL file '{jsonl_file_path}': {str(e)}")
            return {}

    def _parse_complexity(self, complexity: Union[int, str, Tuple[int, int]]) -> Tuple[int, Optional[int]]:
        """
        Parse complexity parameter, supporting single value or range.

        Returns:
            Tuple (min_complexity, max_complexity), where max_complexity is *inclusive* for randint.
            For a single value, max_complexity is the same as min_complexity.
        """
        if isinstance(complexity, int):
            if complexity < 0:
                 raise ValueError("Complexity cannot be negative.")
            return (complexity, complexity) # min and max are the same for fixed value

        elif isinstance(complexity, str) and "-" in complexity:
            try:
                start, end = map(str.strip, complexity.split("-"))
                min_comp = int(start)
                max_comp = int(end)
                if min_comp < 0 or max_comp < 0:
                     raise ValueError("Complexity range values cannot be negative.")
                if min_comp > max_comp:
                    raise ValueError(f"Invalid complexity range string: {complexity}. Start must be <= end.")
                return (min_comp, max_comp)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid complexity range string: {complexity}, should be in 'start-end' integer format. Error: {e}")

        elif isinstance(complexity, (list, tuple)) and len(complexity) == 2:
            try:
                start, end = complexity
                min_comp = int(start)
                max_comp = int(end)
                if min_comp < 0 or max_comp < 0:
                     raise ValueError("Complexity range values cannot be negative.")
                if min_comp > max_comp:
                     raise ValueError(f"Invalid complexity range tuple/list: {complexity}. Start must be <= end.")
                return (min_comp, max_comp)
            except (ValueError, TypeError) as e:
                 raise ValueError(f"Invalid complexity range tuple/list: {complexity}. Both elements must be integers. Error: {e}")

        else:
            raise ValueError(f"Invalid complexity parameter format: {complexity}. Should be integer, 'start-end' string, or tuple/list of two integers.")

    def _get_actual_complexity(self, min_complexity: int, max_complexity: int, available_attributes: int) -> int:
        """
        Get actual complexity value based on complexity range and available attributes.

        Parameters:
            min_complexity: Minimum complexity (inclusive).
            max_complexity: Maximum complexity (inclusive).
            available_attributes: Number of attributes available for the object.

        Returns:
            Actual complexity value, adjusted for available attributes.
        """
        # Determine target complexity within the allowed range
        target_complexity = random.randint(min_complexity, max_complexity)

        # Adjust complexity based on the number of available attributes
        # Ensure complexity is not negative and not more than available attributes
        actual_complexity = max(0, min(target_complexity, available_attributes))
        return actual_complexity

    def _select_random_attributes(self, attributes: Dict[str, List[str]], complexity: int) -> Dict[str, str]:
        """Select random attributes and their values from attribute set"""
        if not attributes or complexity <= 0:
            return {}

        attr_names = list(attributes.keys())
        actual_complexity = min(complexity, len(attr_names)) # Should already be adjusted, but double check

        # Handle the edge case where actual_complexity becomes 0 after adjustment
        if actual_complexity == 0:
            return {}

        selected_attrs = random.sample(attr_names, actual_complexity)
        selected_attr_values = {}

        for attr in selected_attrs:
            values = attributes.get(attr, [])
            if values: # Only include if the attribute has values
                selected_attr_values[attr] = random.choice(values)
            else:
                 # This case shouldn't happen if input data is clean, but good to note
                 print(f"Warning: Attribute '{attr}' has no values. Skipping.")

        # Filter out attributes that ended up with no value (should be rare)
        return {k: v for k, v in selected_attr_values.items() if v}


    async def generate_natural_prompt(self, session: aiohttp.ClientSession,
                                   semaphore: asyncio.Semaphore,
                                   object_name: str,
                                   attributes_used: Dict[str, str],
                                   complexity: int) -> Optional[str]:
        """
        Generate a natural sentence using DeepSeek API via streaming, respecting the semaphore.

        Returns:
            Generated natural sentence prompt or None if generation fails after retries.
        """
        # --- Handle complexity 0 case without API call (remains unchanged) ---
        if complexity == 0:
            if attributes_used:
                 attributes_str_simple = ", ".join(attributes_used.values())
                 return f"A {object_name} that is {attributes_str_simple}."
            else:
                 return f"A {object_name}."

        # --- Prepare Prompt and Payload (remains mostly unchanged, add stream=True) ---
        attributes_str = ", ".join([f"{key}: {value}" for key, value in attributes_used.items()])
        instruction = f"""# Task: Create a Natural Descriptive Sentence
## Input Information
- **Object**: {object_name}
- **Attributes**: {attributes_str}
- **Complexity level**: {complexity}
## Requirements
1. Generate exactly one concise, fluent English sentence describing the **Object**.
2. Naturally incorporate the meaning of all listed attribute values into the sentence.
3. Integrate attribute values smoothly. Avoid simply listing them using phrases like "has," "with," or "featuring."
4. Focus on the attribute *values*, not the attribute *names*. Do not repeat attribute names in the output.
5. Don't add any unnecessary extra modifiers that weren't provided.
6. Do not add any attributes, descriptions, or explanations not present in the input.
7. Avoid unnecessary introductory phrases like "This is..." or "It is...".
8. Strictly adhere to all requirements. Ensure the output is only the single sentence.
## Examples
### Correct Example 1 (Simple Object)
**Input:**
- **Object**: cat
- **Attributes**: color: orange, fur: fluffy, personality: friendly
**Output:**
An orange, fluffy, friendly cat.
### Correct Example 2 (Complex Object Name as Subject)
**Input:**
- **Object**: Ancient Oak Tree
- **Attributes**: age: centuries-old, appearance: gnarled branches, location: forest clearing
**Output:**
A centuries-old ancient oak tree with gnarled branches stands in the forest clearing.
### Incorrect Example 1 (Formulaic Listing)
**Input:**
- **Object**: Sports Car
- **Attributes**: color: red, speed: fast, feature: convertible
**Output (Incorrect):** A sports car with the color red, fast speed, and it is a convertible.
**Output (Desired):** A fast, red convertible sports car.
### Incorrect Example 2 (Adding extra words)
**Input:**
- **Object**: Athlete
- **Attributes**: name: Usain Bolt
**Output:**(Incorrect)** Usain Bolt, the renowned athlete.
**Output (Desired):** Usain Bolt, the athlete.
### Incorrect Example 3 (Adding extra words)
**Input:**
- **Object**: Athlete
- **Attributes**: name: Usain Bolt
**Output:**(Incorrect)** Usain Bolt, the renowned athlete.
**Output (Desired):** Usain Bolt, the athlete.
## Your Response
Return **only** the single generated sentence, without any extra text, introductions, markdown formatting, or quotes surrounding the sentence itself.
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert language assistant skilled at crafting concise, natural English sentences from object names and attributes, strictly following user requirements."},
                {"role": "user", "content": instruction}
            ],
            "temperature": 0.3,
            "max_tokens": 150,
            "stream": True  # <<< Enable streaming
            # "frequency_penalty": 0.1, # Optional
            # "presence_penalty": 0.1 # Optional
        }

        # --- Retry Logic and Semaphore (remains unchanged) ---
        max_retries = 100
        retry_delay = 0.5 # Base delay, will be increased exponentially

        async with semaphore:
             for attempt in range(max_retries):
                 try:
                     timeout = aiohttp.ClientTimeout(total=60, connect=15, sock_read=45)
                     full_response_content = "" # <<< Accumulator for stream content
                     stream_completed_normally = False # <<< Flag to check if '[DONE]' was received

                     # --- Perform API call and process stream ---
                     async with session.post(
                         f"{self.base_url}/chat/completions",
                         headers=self.headers,
                         json=payload,
                         timeout=timeout
                     ) as response:
                         response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

                         # --- Stream processing loop (adapted from DeepSeekObjectGenerator) ---
                         async for chunk in response.content:
                             try:
                                 chunk_text = chunk.decode('utf-8')
                                 # Process potentially multiple data chunks separated by "data: "
                                 if 'data: ' in chunk_text:
                                     json_parts = [part.strip() for part in chunk_text.split('data: ') if part.strip()]

                                     for json_part in json_parts:
                                         if json_part == '[DONE]':
                                             stream_completed_normally = True
                                             break # Exit inner loop once done signal is found

                                         try:
                                             data = json.loads(json_part)
                                             if 'choices' in data and data['choices'] and 'delta' in data['choices'][0]:
                                                 content = data['choices'][0]['delta'].get('content', '')
                                                 if content:
                                                     full_response_content += content # Append content delta
                                         except json.JSONDecodeError:
                                             print(f"Warning: JSON decode error on chunk part: '{json_part}' for '{object_name}' (Attempt {attempt + 1}). Skipping part.")
                                             continue # Skip malformed JSON part

                                     if stream_completed_normally:
                                          break # Exit outer loop if '[DONE]' was received

                             except UnicodeDecodeError:
                                 print(f"Warning: Unicode decode error on chunk for '{object_name}' (Attempt {attempt + 1}). Skipping chunk.")
                                 continue # Skip malformed chunk

                         # --- End of stream processing loop ---

                         if not stream_completed_normally and attempt < max_retries - 1:
                              # If the stream ended without [DONE] (e.g., connection drop), retry.
                              print(f"Warning: Stream for '{object_name}' ended unexpectedly (Attempt {attempt + 1}/{max_retries}). Retrying...")
                              await asyncio.sleep(1) # Small delay before retry
                              continue # Go to next attempt in the outer loop

                         if not full_response_content and stream_completed_normally:
                            # Stream finished but no actual content was received.
                            print(f"Warning: Stream completed for '{object_name}' but no content received (Attempt {attempt + 1}/{max_retries}). Retrying...")
                            # Optional: Add a small delay even on success if API sometimes returns empty streams
                            await asyncio.sleep(1)
                            continue # Try again

                         # --- Process the accumulated result ---
                         generated_text = full_response_content.strip()

                         # Basic cleanup (remains unchanged)
                         if generated_text.startswith('"') and generated_text.endswith('"'):
                             generated_text = generated_text[1:-1]
                         if generated_text.startswith("'") and generated_text.endswith("'"):
                            generated_text = generated_text[1:-1]
                         if generated_text.lower().startswith("output:"):
                              generated_text = generated_text[len("output:"):].strip()

                         # Additional cleanup for escape characters (remains unchanged)
                         generated_text = generated_text.replace('\\"', '"')
                         generated_text = generated_text.replace('\\n', ' ')
                         generated_text = generated_text.replace('\\t', ' ')
                         generated_text = generated_text.replace('\\\\', '\\')

                         # Basic validation (remains unchanged)
                         if not generated_text or len(generated_text) < 3 or "sorry" in generated_text.lower() or "cannot fulfill" in generated_text.lower():
                             print(f"Warning: Received potentially invalid/empty prompt post-stream for '{object_name}' (Attempt {attempt+1}/{max_retries}): '{generated_text}'. Retrying...")
                             await asyncio.sleep(1) # Optional small delay
                             continue # Try again

                         # --- Success ---
                         return generated_text.strip()

                 except (asyncio.TimeoutError, aiohttp.ClientError, aiohttp.ServerDisconnectedError, aiohttp.ClientResponseError) as e:
                     # --- Error Handling and Exponential Backoff (remains unchanged) ---
                     current_delay = retry_delay * (2 ** attempt) # Exponential backoff
                     error_msg = f"{type(e).__name__}"
                     status_code = None
                     if isinstance(e, aiohttp.ClientResponseError):
                          error_msg += f" (Status: {e.status}, Message: '{e.message}')"
                          status_code = e.status
                     elif isinstance(e, asyncio.TimeoutError):
                         error_msg += " (Request timed out)"

                     if attempt < max_retries - 1:
                         print(f"Warning: API call for '{object_name}' (Compl. {complexity}) failed: {error_msg}. Retrying in {current_delay:.1f}s (Attempt {attempt + 1}/{max_retries})...")
                         if status_code == 429: # Rate limited
                              print(f"Info: Rate limit hit for '{object_name}'. Consider lowering concurrent_limit.")
                              current_delay = max(current_delay, 10)
                         elif status_code and 500 <= status_code <= 599: # Server error
                              print(f"Info: Server error ({status_code}) for '{object_name}'.")
                              current_delay = max(current_delay, 5)

                         await asyncio.sleep(current_delay)
                     else:
                         print(f"Error: Failed to generate prompt for '{object_name}' after {max_retries} attempts due to {error_msg}. Skipping.")
                         return None # Indicate failure

                 except Exception as e:
                     # Catch other unexpected errors (remains unchanged)
                     print(f"Error: Unexpected error during prompt generation for '{object_name}': {type(e).__name__}: {str(e)}. Skipping.")
                     traceback.print_exc()
                     return None # Indicate failure

        # Fallthrough case: exhausted retries (remains unchanged)
        print(f"Error: Prompt generation failed for '{object_name}' after {max_retries} retries.")
        return None


    async def _process_single_prompt_task(self,
                                          job_info: Dict[str, Any],
                                          session: aiohttp.ClientSession,
                                          semaphore: asyncio.Semaphore,
                                          output_file_handle: Optional[Any]) -> Optional[Dict[str, Any]]:
        """Helper coroutine to process one prompt generation job."""
        object_name = job_info["object_name"]
        attributes_used = job_info["attributes_used"]
        actual_complexity = job_info["actual_complexity"]
        category = job_info["category"]
        category_id = job_info["category_id"]
        object_id = job_info["object_id"] # This is the temp ID within category
        prompt_index = job_info["prompt_index"]

        # This now calls the modified stream-based generate_natural_prompt
        prompt_text = await self.generate_natural_prompt(
            session, semaphore, object_name, attributes_used, actual_complexity
        )

        if prompt_text is None:
             # Failure already logged in generate_natural_prompt
             return None # Indicate failure for this task

        # If successful, prepare the final data structure
        # Safely acquire the next global prompt ID and increment the counter
        async with self.file_lock: # Use lock to ensure unique, sequential IDs and safe file write
            current_total_prompt_id = self.total_prompt_id_counter
            self.total_prompt_id_counter += 1

            prompt_data = {
                "object_id": object_id, # Original temp ID from input load
                "prompt_id": prompt_index,  # Prompt index (1 to N) for this specific object
                "total_prompt_id": current_total_prompt_id, # Globally unique sequential ID
                "complexity": actual_complexity,
                "category": category,
                "category_id": category_id,
                "object": object_name,
                "prompt": prompt_text,
                "attributes_used": attributes_used
            }

            # Stream write to file immediately if file handle exists
            if output_file_handle:
                try:
                    output_file_handle.write(json.dumps(prompt_data, ensure_ascii=False) + '\n')
                    # output_file_handle.flush() # Consider flushing less often
                except Exception as e:
                     print(f"Error: Failed to write prompt {current_total_prompt_id} for '{object_name}' to file: {e}")
                     # return None # Option: Consider write failure as task failure

        return prompt_data # Return data whether writing to file or not

    async def generate_all_prompts(self,
                                complexity: Union[int, str, Tuple[int, int]] = 5,
                                prompts_per_object: int = 1,
                                output_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate prompts for all objects concurrently and save them in real-time if specified.

        Parameters:
            complexity: Complexity target. Integer, "min-max" string, or (min, max) tuple.
            prompts_per_object: Number of prompts to generate per object.
            output_file: Optional path to output JSONL file. Results streamed if provided.

        Returns:
            List of generated prompts (if output_file is None) or empty list (if streaming to file).
        """
        start_time = time.time()
        all_prompts_data: List[Dict[str, Any]] = [] # Only populated if output_file is None
        output_file_handle = None
        total_prompts_generated = 0
        total_prompts_failed = 0
        initially_skipped_objects = 0 # Track objects skipped before task creation

        if prompts_per_object < 1:
            raise ValueError(f"Prompts per object must be at least 1, received: {prompts_per_object}")

        min_complexity, max_complexity = self._parse_complexity(complexity)

        if min_complexity == max_complexity:
            print(f"Starting prompt generation with fixed complexity: {min_complexity}")
        else:
            print(f"Starting prompt generation with complexity range: {min_complexity}-{max_complexity} (inclusive)")
        print(f"Generating {prompts_per_object} prompt(s) per object.")
        print(f"Concurrency limit: {self.concurrent_limit}")
        print(f"Using API Base URL: {self.base_url}")
        print(f"Using Model: {self.model}")

        # --- 1. Prepare all job descriptions ---
        prompt_jobs = []
        total_potential_prompts = 0
        objects_processed_for_jobs = 0
        print("Preparing generation jobs...")
        for category, objects in self.objects_data.items():
            category_id = self.category_id_map[category]
            for obj in objects:
                objects_processed_for_jobs += 1
                object_id = obj["temp_cat_obj_id"] # Use the temp ID assigned during load
                object_name = obj["name"]
                attributes = obj.get("attributes", {})
                available_attr_count = len(attributes.keys())

                # Skip object entirely if it has no attributes and minimum complexity requires them
                if available_attr_count == 0 and min_complexity > 0:
                    initially_skipped_objects += prompts_per_object
                    continue # Skip this object

                for i in range(prompts_per_object):
                    total_potential_prompts += 1 # Count potential prompts before filtering
                    actual_complexity = self._get_actual_complexity(min_complexity, max_complexity, available_attr_count)
                    attributes_used = self._select_random_attributes(attributes, actual_complexity)

                    # Skip this specific prompt if complexity > 0 but no attributes could be selected
                    if actual_complexity > 0 and not attributes_used and available_attr_count > 0:
                         initially_skipped_objects += 1
                         continue # Skip this specific prompt instance

                    job_info = {
                        "object_name": object_name,
                        "attributes_used": attributes_used,
                        "actual_complexity": actual_complexity,
                        "category": category,
                        "category_id": category_id,
                        "object_id": object_id,
                        "prompt_index": i + 1 # 1-based index for prompts per object
                    }
                    prompt_jobs.append(job_info)


        if not prompt_jobs:
            print("Warning: No prompt generation jobs could be created based on input data and parameters.")
            if objects_processed_for_jobs > 0:
                 print(f"Info: {initially_skipped_objects} potential prompts were skipped due to missing attributes or selection issues.")
            return []

        print(f"Prepared {len(prompt_jobs)} actual prompt generation jobs (out of {total_potential_prompts} potential).")
        if initially_skipped_objects > 0:
             print(f"Info: {initially_skipped_objects} potential prompts were skipped before task creation.")


        # --- 2. Setup file writing and session ---
        if output_file:
            try:
                # Use 'w' mode to ensure a fresh file at the start
                output_file_handle = open(output_file, 'w', encoding='utf-8')
                print(f"Streaming results to '{output_file}'...")
            except Exception as e:
                print(f"Error: Could not open output file '{output_file}' for writing: {str(e)}. Aborting.")
                return [] # Stop if we can't open the file as requested

        # Use a semaphore to limit concurrent API calls
        semaphore = asyncio.Semaphore(self.concurrent_limit)
        # Setup persistent session
        conn = aiohttp.TCPConnector(limit=self.concurrent_limit * 2, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30) # Use task-specific timeouts for requests

        try:
             async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                # --- 3. Create and run tasks ---
                tasks = []
                for job_info in prompt_jobs:
                    task = asyncio.create_task(
                        self._process_single_prompt_task(
                            job_info, session, semaphore, output_file_handle
                        )
                    )
                    tasks.append(task)

                print(f"Launching {len(tasks)} generation tasks...")
                results = await tqdm_asyncio.gather(
                    *tasks,
                    desc="Generating Prompts",
                    total=len(tasks),
                    unit="task"
                )

                # --- 4. Process results ---
                print("\nProcessing results...") # Add separator
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        task_name = tasks[i].get_name() if hasattr(tasks[i], 'get_name') else f"Task_{i}"
                        print(f"Error: Task {task_name} failed with exception: {result}")
                        total_prompts_failed += 1
                    elif result is None:
                        total_prompts_failed += 1
                    elif isinstance(result, dict):
                        total_prompts_generated += 1
                        if not output_file_handle: # Only append if not writing to file
                            all_prompts_data.append(result)
                    else:
                        print(f"Warning: Unexpected result type from task {i}: {type(result)}")
                        total_prompts_failed += 1

        except Exception as e:
            print(f"\nFATAL Error during concurrent processing setup or gathering: {type(e).__name__}: {e}")
            traceback.print_exc() # Print stack trace for fatal errors outside task execution
            if output_file_handle and not output_file_handle.closed:
                 output_file_handle.close()
                 print("Output file closed due to error.")
            return [] # Return empty on fatal error during processing
        finally:
            # --- 5. Cleanup ---
            if output_file_handle and not output_file_handle.closed:
                output_file_handle.flush() # Ensure buffer is written
                output_file_handle.close()
                print(f"\nFinished streaming to '{output_file}'.")

        elapsed_time = time.time() - start_time
        print("\n--- Prompt Generation Summary ---")
        print(f"Total potential prompts considered: {total_potential_prompts}")
        print(f"Prompts skipped before task creation: {initially_skipped_objects}")
        print(f"Tasks launched: {len(prompt_jobs)}")
        print(f"Successfully generated: {total_prompts_generated}")
        print(f"Failed during generation: {total_prompts_failed}")
        if len(prompt_jobs) != (total_prompts_generated + total_prompts_failed):
             print(f"Warning: Task count mismatch! Launched={len(prompt_jobs)}, Success={total_prompts_generated}, Failed={total_prompts_failed}")
        print(f"Final global prompt ID assigned: {self.total_prompt_id_counter - 1}")
        print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")

        if elapsed_time > 0:
             prompts_per_sec = total_prompts_generated / elapsed_time
             print(f"Average generation rate: {prompts_per_sec:.2f} prompts/second")

        if not output_file_handle:
            all_prompts_data.sort(key=lambda x: x.get('total_prompt_id', 0))
            return all_prompts_data
        else:
            if output_file and os.path.exists(output_file):
                 try:
                     with open(output_file, 'r', encoding='utf-8') as f_check:
                          lines = sum(1 for _ in f_check)
                     print(f"Verification: Output file '{output_file}' contains {lines} lines.")
                     if lines != total_prompts_generated:
                          print(f"Warning: Line count ({lines}) in output file does not match expected generated count ({total_prompts_generated}).")
                 except Exception as e:
                      print(f"Warning: Could not verify output file content: {e}")

            return []
# if __name__ == "__main__":
    # api_key_file = "/home/ubuntu/dcai/deepseek_api_key.txt"
#     # If you run this file directly, just use the synchronous interface
#     # No need to manually call asyncio.run(main())
    
#     # # Initialize the generator
#     # generator = DeepSeekObjectGenerator(
#     #     api_key_file=api_key_file,  # Replace with the path to your API key file
#     #     default_objects_per_category=2,  # Optional: set the default number of objects to generate per category
#     #     concurrent_limit=3  # Optional: set the maximum number of concurrent requests
#     # )

#     #    # Method 2: Customize the number of objects generated for each category
#     # custom_counts = {
#     #     "Natural Landscapes" : 1,
#     #     "Cities and Architecture" : 1,
#     #     "People" : 2,
#     #     "Animals" : 2,
#     #     "Plants" : 2,
#     #     "Food and Beverages" : 2,
#     #     "Sports and Fitness" : 2,
#     #     "Art and Culture" : 2,
#     #     "Technology and Industry" : 2,
#     #     "Everyday Objects" : 2,
#     #     "Transportation" : 2,
#     #     "Abstract and Conceptual Art" : 2
#     #     # Other categories will use the default value
#     # }
    
#     # # # Generate objects and save to file
#     # generator.generate_and_save_objects_stream(output_file="objects.jsonl")


    # # Create manager instance
    # manager = DeepSeekAttributeManager(
    #     api_key_file=api_key_file,  # File containing the API key
    #     concurrent_limit=5,          # Number of concurrent requests
    #     values_per_attribute=5       # Number of values to generate per attribute
    # )

    # # One-stop processing: generate attributes and values for attributes
    # manager.process_all(
    #     input_file="objects.jsonl",
    #     output_file="attributes_library.jsonl"
    # )

#     generator = PromptGenerator(
#         jsonl_file_path="attributes_library.jsonl",
#         api_key_file=api_key_file
#     )

#     # # # Generate 3 prompts for each object with a fixed complexity of 2
#     # # asyncio.run(generator.generate_all_prompts(
#     # #     complexity=2,
#     # #     prompts_per_object=2,  # Generate 3 prompts per object
#     # #     output_file="multi_prompts_fixed_complexity.jsonl"
#     # # ))
    
    
#        # Generate 2 prompts per object with complexity range 0-4
#     asyncio.run(generator.generate_all_prompts(
#         complexity="0-11",  # Complexity range 0 to 3 (inclusive left, exclusive right)
#         prompts_per_object=5,  # Generate 2 prompts per object
#         output_file="prompts.jsonl"
#     ))
