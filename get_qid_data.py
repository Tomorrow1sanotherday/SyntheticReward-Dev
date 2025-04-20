# get_qid_data.py
from vqa_feedback import load_all_qid_data, process_and_save_prompt
from util.data_process import read_only_prompts_from_jsonl
import argparse
import sys
import concurrent.futures
import os
import time

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate qid_data.json with concurrency and streaming writes")
    parser.add_argument("--input_file", type=str, default="asset/aaa.jsonl", help="Input file path")
    parser.add_argument("--output_file", type=str, default="qid_data.json", help="Output file path")
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")
    args = parser.parse_args()

    prompts_file = args.input_file
    output_file = args.output_file
    max_workers = args.max_workers

    # 1. 加载现有数据
    print(f"Loading existing data from {output_file}...")
    existing_data_list = load_all_qid_data(output_file)
    existing_prompts = {item['prompt'] for item in existing_data_list}
    print(f"Loaded {len(existing_data_list)} existing entries.")

    # 2. 加载所有提示
    print(f"Loading prompts from {prompts_file}...")
    all_prompts = read_only_prompts_from_jsonl(prompts_file)
    print(f"Loaded {len(all_prompts)} prompts.")

    # 3. 确定需要处理的提示
    prompts_to_process = [prompt for prompt in all_prompts if prompt not in existing_prompts]
    print(f"Found {len(prompts_to_process)} prompts to process.")

    if not prompts_to_process:
        print("No new prompts to process. Exiting.")
        return 0

    # 4. 使用 ThreadPoolExecutor 并发处理并实时保存
    print(f"Starting concurrent processing with streaming writes, {max_workers} workers for {len(prompts_to_process)} prompts...")
    
    # 记录开始时间以计算总耗时和速度
    start_time = time.time()
    
    # 创建一个共享的数据列表，用于存储所有数据
    # 使用现有数据初始化它
    shared_data_list = existing_data_list
    
    # 用于跟踪进度的计数器
    success_count = 0
    error_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建一个任务列表，每个任务处理一个提示并实时保存结果
        futures = [executor.submit(
            process_and_save_prompt, 
            prompt, 
            shared_data_list, 
            output_file,
            max_retries=100,     # 添加最大重试次数
            retry_delay=0      # 添加重试延迟（秒）
        ) for prompt in prompts_to_process]
        
        # 处理完成的结果
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                success, prompt = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                
                # 计算并显示进度和预估剩余时间
                elapsed_time = time.time() - start_time
                processed = success_count + error_count
                if processed > 0:
                    avg_time_per_prompt = elapsed_time / processed
                    remaining_prompts = len(prompts_to_process) - processed
                    estimated_time_remaining = avg_time_per_prompt * remaining_prompts
                    
                    # 计算处理速度（每分钟处理的提示数）
                    prompts_per_minute = (processed / elapsed_time) * 60 if elapsed_time > 0 else 0
                    
                    print(f"Progress: {processed}/{len(prompts_to_process)} ({processed/len(prompts_to_process)*100:.1f}%) | "
                          f"Success: {success_count} | Errors: {error_count} | "
                          f"Speed: {prompts_per_minute:.1f} prompts/min | "
                          f"Est. remaining: {estimated_time_remaining/60:.1f} minutes")
            except Exception as e:
                error_count += 1
                print(f"Error handling future result: {e}")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"Processing complete in {total_time/60:.1f} minutes.")
    print(f"Final results: {success_count} successful, {error_count} errors out of {len(prompts_to_process)} prompts.")
    print(f"Average speed: {(success_count + error_count) / total_time * 60:.1f} prompts per minute.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
