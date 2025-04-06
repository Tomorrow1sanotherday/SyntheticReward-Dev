from vqa_feedback import generate_qid_data_from_prompt
from util.data_process import read_only_prompts_from_jsonl

prompts_file = 'asset/simple_prompts.jsonl'
prompts = read_only_prompts_from_jsonl(prompts_file)

for i in range(len(prompts)):
    print(generate_qid_data_from_prompt(prompts[i]))