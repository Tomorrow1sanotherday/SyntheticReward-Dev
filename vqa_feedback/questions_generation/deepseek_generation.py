from PIL import Image
from .query_utils import generate_dsg
import openai
from .deepseek_utils import deepseek_setup, deepseek_completion
from .parse_utils import parse_tuple_output, parse_dependency_output, parse_question_output
import json
import os

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

def load_qid_data_from_json(prompt, filename='qid_data.json'):
    """
    从 JSON 文件加载 qid_data。
    """
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        for item in existing_data:
            if item['prompt'] == prompt:
                return item
    return None

def generate_qid_data_from_prompt(input_text_prompt):
    """
    从提示生成问题数据。
    """
    deepseek_setup('./deepseek_api_key.txt')
    # 确保在调用之前已设置 API 密钥
    assert openai.api_key is not None, "API key not found. Please ensure it's set in deepseek_setup."

    qid_data = load_qid_data_from_json(input_text_prompt)

    if qid_data:
        print(f"Data for prompt '{input_text_prompt}' found in JSON file. Returning existing data.")
        return qid_data

    id2prompts = {'custom_0': {'input': input_text_prompt}}

    id2tuple_outputs, id2question_outputs, id2dependency_outputs = generate_dsg(
        id2prompts,
        generate_fn=lambda prompt: deepseek_completion(prompt, model="deepseek-chat", temperature=0, max_tokens=500)
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

    save_qid_data_to_json(qid_data)

    return qid_data