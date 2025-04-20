import json

def jsonl2txt(input_file, output_file):
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)  # 使用 json.loads 解析每一行
            prompts.append(data['prompt'])  # 假设每行数据中有 'prompt' 字段
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(prompt + '\n')  # 将所有 prompts 写入输出文件

if __name__ == '__main__':
    jsonl2txt('/home/ubuntu/dcai/SyntheticReward-Dev/asset/simple_prompts.jsonl', '/home/ubuntu/dcai/SyntheticReward-Dev/d3po-main/d3po_pytorch/assets/simple_prompt.txt')
    
