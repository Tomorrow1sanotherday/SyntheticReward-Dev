import json

# with open('qid_data_simple.json') as f:
#     data = json.load(f)
# print(len(data))
# import json
# from collections import Counter

# # 假设文件名是 'data.jsonl'
# file_path = '/home/ubuntu/dcai/SyntheticReward-Dev/asset/simple_prompts.jsonl'

# # 用于存储所有 prompt 的列表
# prompts = []

# # 逐行读取文件并提取 prompt
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         # 解析每一行的 JSON 数据
#         data = json.loads(line.strip())  # .strip() 用于去除行尾换行符
#         prompts.append(data['prompt'])  # 提取 prompt 字段并添加到列表

# # 统计每个 prompt 出现的次数
# prompt_counts = Counter(prompts)

# # 输出所有重复的 prompt 以及它们的出现次数
# for prompt, count in prompt_counts.items():
#     if count > 1:
#         print(f"Prompt: '{prompt}', Count: {count}")


# import json

# # 读取 JSON 数据
# with open('qid_data_simple.json', "r", encoding='utf-8') as f:
#     data = json.load(f)


# # 用于统计不一致的条目
# mismatch_count = 0

# # 遍历每个对象，检查字段内容数量是否一致
# for item in data:
#     prompt = item['prompt']
#     qid2tuple_len = len(item['qid2tuple'])
#     qid2dependency_len = len(item['qid2dependency'])
#     qid2question_len = len(item['qid2question'])
    
#     if qid2tuple_len != qid2dependency_len or qid2tuple_len != qid2question_len:
#         print(f"Mismatch found in prompt: {prompt}")
#         mismatch_count += 1

# print(f"\nTotal mismatches found: {mismatch_count}")

import os
print(os.isdir)