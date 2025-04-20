import json
import random

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


def shuffle_jsonl_file(input_filepath, output_filepath, encoding='utf-8'):
    """
    读取一个 JSONL 文件，随机打乱其行的顺序，并将结果写入一个新的 JSONL 文件。

    Args:
        input_filepath (str): 输入的 JSONL 文件路径。
        output_filepath (str): 输出打乱后内容的 JSONL 文件路径。
        encoding (str, optional): 读取和写入文件时使用的字符编码。默认为 'utf-8'。

    Raises:
        FileNotFoundError: 如果输入文件未找到。
        Exception: 处理过程中可能出现的其他 I/O 错误或异常。

    注意:
        此函数会将整个输入文件读入内存。对于超出可用内存的超大文件，
        可能不适用。
    """
    try:
        # 1. 从输入文件中读取所有行
        print(f"正在从 '{input_filepath}' (编码: {encoding}) 读取行...")
        with open(input_filepath, 'r', encoding=encoding) as infile:
            lines = infile.readlines() # readlines() 保留行尾的换行符
        print(f"已读取 {len(lines)} 行。")

        # 处理空文件的情况
        if not lines:
            print("输入文件为空，无需打乱。将创建一个空的输出文件。")
            # 创建一个空的输出文件以保持行为一致性
            with open(output_filepath, 'w', encoding=encoding) as outfile:
                pass # 直接创建一个空文件
            print(f"已创建空的输出文件: '{output_filepath}'")
            return # 函数执行完毕

        # 2. 打乱行列表
        print("正在打乱行顺序...")
        random.shuffle(lines)
        print("打乱完成。")

        # 3. 将打乱后的行写入输出文件
        print(f"正在将打乱后的行写入 '{output_filepath}' (编码: {encoding})...")
        with open(output_filepath, 'w', encoding=encoding) as outfile:
            outfile.writelines(lines) # writelines 将列表中的所有字符串写入文件
        print("写入完成。")
        print(f"成功创建打乱后的文件: '{output_filepath}'")

    except FileNotFoundError:
        print(f"错误: 输入文件未找到 '{input_filepath}'")
        # 你可以选择在这里处理错误，或者重新抛出让调用者处理
        raise # 重新抛出 FileNotFoundError

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        # 同样，可以选择处理或重新抛出
        raise # 重新抛出其他异常

if __name__ == "__main__":
    # simple_prompts = '/home/ubuntu/dcai/SyntheticReward-Dev/prompt_generator/asset/simple_prompts.jsonl'
    # list1 = read_only_prompts_from_jsonl(simple_prompts)
    # print(len(list1))
     # --- 配置 ---
    # !!! 将 'your_input.jsonl' 替换为你的实际输入文件名 !!!
    input_jsonl_file = 'asset/hard_prompts.jsonl'
    # !!! 将 'shuffled_output.jsonl' 替换为你想要的输出文件名 !!!
    output_jsonl_file = 'asset/hard_prompts_shuffle.jsonl'
    # 如果你的文件编码不是 utf-8，请修改这里
    file_encoding = 'utf-8'

    print("--- 开始执行 JSONL 文件打乱脚本 ---")
    try:
        # 调用函数来执行打乱操作
        # 在调用前，确保你已经创建了一个名为 'your_input.jsonl' 的文件
        # 包含你的 JSONL 数据，或者修改上面的 input_jsonl_file 变量
        print(f"尝试打乱 '{input_jsonl_file}' 并保存到 '{output_jsonl_file}'...")
        shuffle_jsonl_file(input_jsonl_file, output_jsonl_file, encoding=file_encoding)
        print("--- 脚本成功执行完毕 ---")

    except FileNotFoundError:
        # 函数内部已经打印了错误信息，这里可以添加额外的处理或提示
        print(f"--- 脚本执行失败: 请确保输入文件 '{input_jsonl_file}' 存在于当前目录。 ---")
    except Exception as e:
        # 捕获其他可能的错误
        print(f"--- 脚本执行过程中发生未预料的错误: {e} ---")
