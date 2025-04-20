# deepseek_generation.py
from PIL import Image
from .query_utils import generate_dsg
import openai
from .deepseek_utils import deepseek_setup, deepseek_completion
from .parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
import json
import os
import threading
import time

# 创建一个文件锁，用于安全地写入共享文件
file_lock = threading.Lock()

def load_qid_data_from_json(prompt, filename='qid_data.json'):
    """
    从 JSON 文件加载 qid_data。如果文件为空，返回 None。
    """
    if os.path.exists(filename):
        # 检查文件是否为空
        if os.path.getsize(filename) == 0:
            return None
        
        # 文件不为空，加载 JSON 数据
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.decoder.JSONDecodeError:
                # 如果 JSON 格式无效，返回 None
                return None

        # 查找匹配的 prompt
        for item in existing_data:
            if item['prompt'] == prompt:
                return item
    return None

def generate_qid_data_from_prompt(input_text_prompt, output_file):
    """
    从提示生成问题数据。
    """

    qid_data = load_qid_data_from_json(input_text_prompt, output_file)

    if qid_data:
        print(f"Data for prompt '{input_text_prompt}' found in JSON file. Returning existing data.")
        return qid_data

    id2prompts = {'custom_0': {'input': input_text_prompt}}

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        generate_fn=lambda prompt: deepseek_completion(prompt, model="deepseek-v3", temperature=0, max_tokens=500)
    )

    qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
    qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
    qid2question = parse_question_output(id2question_outputs['custom_0']['output'])

    qid_data = {
        'prompt': input_text_prompt,
        'qid2tuple': qid2tuple,
        'qid2dependency': qid2dependency,
        'qid2question': qid2question
    }

    save_qid_data_to_json(qid_data, output_file)

    return qid_data

def save_qid_data_to_json(qid_data, filename='qid_data.json'):
    """
    将 qid_data 保存到 JSON 文件，如果相同的提示不存在。
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    prompt_exists = any(item['prompt'] == qid_data['prompt'] for item in existing_data)

    if not prompt_exists:
        existing_data.append(qid_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
        print(f"Data with prompt '{qid_data['prompt']}' has been written to {filename}.")
    else:
        print(f"Data with prompt '{qid_data['prompt']}' already exists. No data written.")

def load_all_qid_data(filename='qid_data.json'):
    """
    加载 JSON 文件中的所有 qid_data。如果文件不存在、为空或无效，返回空列表。
    """
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
        return []
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Warning: {filename} contains non-list data. Returning empty list.")
                return []
            return data
    except json.decoder.JSONDecodeError:
        print(f"Error decoding JSON from {filename}. File might be corrupt. Returning empty list.")
        return []
    except Exception as e:
        print(f"An error occurred while loading {filename}: {e}. Returning empty list.")
        return []

def save_all_qid_data(data, filename='qid_data.json'):
    """
    将整个 qid_data 列表保存到 JSON 文件。
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"An error occurred while saving data to {filename}: {e}")
        return False

def append_qid_data_with_lock(qid_data, data_list, filename='qid_data.json'):
    """
    使用锁安全地将单个 qid_data 添加到内存中的列表，并将更新后的整个列表写入文件。
    返回 True 表示成功，False 表示失败。
    """
    result = False
    with file_lock:
        # 检查 prompt 是否已经存在
        if any(item['prompt'] == qid_data['prompt'] for item in data_list):
            print(f"Prompt '{qid_data['prompt']}' already exists in memory. Not adding duplicate.")
            return True  # 返回 True 因为数据已经存在，不需要添加
        
        # 添加到内存中的列表
        data_list.append(qid_data)
        
        # 写入文件
        result = save_all_qid_data(data_list, filename)
        if result:
            print(f"Data for prompt '{qid_data['prompt']}' successfully written to {filename}")
        else:
            print(f"Failed to write data for prompt '{qid_data['prompt']}' to {filename}")
    
    return result

def validate_qid_data(qid2tuple, qid2dependency, qid2question):
    """
    验证qid2tuple、qid2dependency、qid2question三个字段的数量是否一致且ID对应
    
    返回:
    bool: 如果三个字段的数量一致且ID对应则返回True，否则返回False
    """
    # 检查是否为空
    if not qid2tuple or not qid2dependency or not qid2question:
        print("Validation failed: One or more of the dictionaries is empty")
        return False
    
    # 检查数量是否一致
    if len(qid2tuple) != len(qid2dependency) or len(qid2tuple) != len(qid2question):
        print(f"Validation failed: Length mismatch - tuple: {len(qid2tuple)}, dependency: {len(qid2dependency)}, question: {len(qid2question)}")
        return False
    
    # 检查ID是否对应
    for qid in qid2tuple:
        if qid not in qid2dependency or qid not in qid2question:
            print(f"Validation failed: ID {qid} is not present in all three dictionaries")
            return False
    
    print(f"Validation passed: All {len(qid2tuple)} IDs are present in all three dictionaries")
    return True

def process_prompt_api_only(input_text_prompt, max_retries=100):
    """
    调用API并解析结果，验证三个部分的数量一致性，不一致则重试。
    """
    retry_count = 0
    while retry_count < max_retries:
        try:
            id2prompts = {'custom_0': {'input': input_text_prompt}}

            id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
                id2prompts,
                generate_fn=lambda prompt: deepseek_completion(prompt, model="deepseek-v3", temperature=0, max_tokens=500),
                verbose=False  # 减少输出
            )

            qid2tuple = parse_tuple_output(id2tuple_outputs['custom_0']['output'])
            qid2dependency = parse_dependency_output(id2dependency_outputs['custom_0']['output'])
            qid2question = parse_question_output(id2question_outputs['custom_0']['output'])

            # 验证三个部分是否一致
            if validate_qid_data(qid2tuple, qid2dependency, qid2question):
                qid_data = {
                    'prompt': input_text_prompt,
                    'qid2tuple': qid2tuple,
                    'qid2dependency': qid2dependency,
                    'qid2question': qid2question
                }
                print(f"Successfully processed prompt after {retry_count+1} attempt(s): '{input_text_prompt}'")
                return qid_data
            else:
                retry_count += 1
                print(f"Inconsistent data, retrying ({retry_count}/{max_retries})...")
                
                # 如果达到最大重试次数，使用最后一次生成的结果
                if retry_count == max_retries:
                    print(f"Reached maximum retries ({max_retries}). Using last generated result.")
                    qid_data = {
                        'prompt': input_text_prompt,
                        'qid2tuple': qid2tuple,
                        'qid2dependency': qid2dependency,
                        'qid2question': qid2question
                    }
                    return qid_data
                
        except Exception as e:
            retry_count += 1
            print(f"Error in attempt {retry_count}: {e}")
            if retry_count >= max_retries:
                raise RuntimeError(f"Failed to process prompt after {max_retries} attempts: {e}")
            time.sleep(0)  # 短暂延迟后重试

def process_and_save_prompt(input_text_prompt, shared_data_list, output_file, max_retries=100, retry_delay=0):
    """
    处理提示并立即保存结果。这个函数被设计为在线程池中运行。
    添加了重试逻辑，在遇到错误时会重试直到成功。
    """
    retry_count = 0
    while True:
        try:
            # 处理 API 调用
            qid_data = process_prompt_api_only(input_text_prompt, max_retries=5)  # 内部验证逻辑会重试直到得到一致的结果
            
            # 安全地添加到内存列表并写入文件
            success = append_qid_data_with_lock(qid_data, shared_data_list, output_file)
            
            if success:
                return True, input_text_prompt
            else:
                print(f"Failed to save data for prompt '{input_text_prompt}'. Retrying ({retry_count + 1}/{max_retries})...")
                retry_count += 1
                if retry_count >= max_retries:
                    print(f"Reached maximum retries ({max_retries}) for saving data. Giving up.")
                    return False, input_text_prompt
                time.sleep(retry_delay)  # 等待一段时间后重试保存
        except Exception as e:
            retry_count += 1
            print(f"Error processing prompt '{input_text_prompt}' (Attempt {retry_count}): {e}")
            if retry_count >= max_retries:
                print(f"Reached maximum retries ({max_retries}). Giving up.")
                return False, input_text_prompt
            
            # 计算退避时间（每次失败后增加等待时间）
            backoff_time = retry_delay * (2 ** (retry_count - 1))  # 指数退避
            print(f"Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
