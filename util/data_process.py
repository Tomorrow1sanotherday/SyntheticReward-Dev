import json

def read_only_prompts_from_jsonl(file_path):
    """
    只读取jsonl文件中的prompt字段并存储到列表中
    
    Args:
        file_path: jsonl文件的路径
        
    Returns:
        prompts_only: 只包含prompt文本的列表
    """
    prompts_only = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    # 只添加prompt字段到列表
                    if 'prompt' in json_obj:
                        prompts_only.append(json_obj['prompt'])
                except json.JSONDecodeError:
                    print(f"警告：无法解析行: {line}")
                    continue
        
        return prompts_only
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []

if __name__ == "__main__":
    simple_prompts = '/home/ubuntu/dcai/SyntheticReward-Dev/prompt_generator/asset/simple_prompts.jsonl'
    list1 = read_only_prompts_from_jsonl(simple_prompts)
    print(len(list1))
