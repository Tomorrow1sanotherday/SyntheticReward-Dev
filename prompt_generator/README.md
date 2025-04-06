# 使用说明
修改 config.yml 中的 base_url 和 api_key_fill。
1. 生成objects

执行脚本：python prompt_generator/examples/generate_objects.py  --objects_count 42 --concurrent_limit 300 --output_file objects.jsonl
参数说明：
objects_count：每一个类别生成的object数量。
concurrent_limit：并行数量。
output_file：输出文件名。
query次数：42*12=504次

2. 生成attributes_library

执行脚本：
python prompt_generator/examples/generate_attributes.py  --attributes_count 12 --input_file objects.jsonl --output_file attributes_library.jsonl --concurrent_limit 300
 参数说明：
 attributes_count：每一个object生成的属性数量.
 input_file：输入的objects文件。
 output_file：输出文件。
 concurrent_limit：并行数量。
 query次数：502（属性概念query次数） + 502 * 12（属性值query次数）= 6526次

3. 生成prompts：
 
 执行脚本：
 生成简单prompt：
 python prompt_generator/examples/generate_prompts.py  --complexit 1-3 --input_file attributes_library.jsonl --output_file simple_prompts.jsonl --concurrent_limit 300 --prompts_count 10
 生成复杂prompt：
  python prompt_generator/examples/generate_prompts.py  --complexit 4-10 --input_file attributes_library.jsonl --output_file hard_prompts.jsonl --concurrent_limit 300 --prompts_count 10
 参数说明：
 complexit： 生成的prompt的复杂度，左闭右闭，0表示什么属性都没有只有单个object。
 input_file：输入的attributes_library文件。
 output_file：输出文件。
 concurrent_limit：并行数量。
 prompts_count：每个object生成的prompt的数量。
 queryc次数：504*10=5040


