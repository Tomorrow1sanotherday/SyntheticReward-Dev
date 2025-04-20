# deepseek_utils.py
import openai
import os
import time

# 从文件加载 API 密钥
def load_api_key(key_path='./deepseek_api_key.txt'):
    if os.path.exists(key_path):
        with open(key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        # 当文件不存在时，抛出更详细的错误，包含当前工作目录
        current_dir = os.getcwd()
        raise FileNotFoundError(f"API key file not found: '{key_path}'. Looked in '{current_dir}'. Please ensure the API key file exists at this path relative to where you run the script.")

# 设置 OpenAI API 密钥和基础 URL
# 注意：deepseek_completion 内部会使用 load_api_key 和显式 base_url，
# 因此这里的设置对于 deepseek_completion 本身可能不是必需的，
# 但作为全局配置仍然保留，以防其他地方依赖。
def deepseek_setup(key_path='./deepseek_api_key.txt'):
    # 可以在这里选择是否调用 load_api_key，或者只设置全局变量
    # 为了避免重复读取文件，可以先检查 openai.api_key 是否已设置
    # if openai.api_key is None: # 这种检查对于新版本的openai库可能不总是有效
    #     api_key = load_api_key(key_path)
    #     openai.api_key = api_key
    #     openai.base_url = 'https://api.nuwaapi.com/v1' # 设置 DeepSeek 的 API 基础 URL
    #     print("API key has been set via deepseek_setup.")
    # else:
    #     print("API key already seems set. Skipping deepseek_setup.")

    # 考虑到 deepseek_completion 内部自己加载key并创建client，
    # deepseek_setup 的主要作用是检查文件是否存在，并作为信号表明配置过程。
    # 我们可以简化 deepseek_setup，只做检查和打印，实际配置发生在 deepseek_completion
    # 或者让 deepseek_completion 完全依赖这里的全局设置。
    # 保持现有结构，让 deepseek_completion 自己加载key更线程安全。
    # 那么 deepseek_setup 主要用于 early check 和全局提示。
    
    # 确保文件存在，如果不存在会抛出 FileNotFoundError
    load_api_key(key_path) 
    # 尽管deepseek_completion会自己用key创建客户端，
    # 这里仍然可以设置全局的，以防其他地方使用。
    # 但更稳妥的做法是让deepseek_completion完全依赖显式传入或内部加载的key，
    # 忽略这里的全局设置，而 deepseek_utils 的现有代码就是这样做的。
    
    # 所以， deepseek_setup 核心意义在于提示和检查文件存在。
    # 如果想让 deepseek_completion 依赖全局设置，需要修改 deepseek_completion。
    # 为了保持现有 deepseek_completion 的线程安全性（每次新建client），
    # 我们让 deepseek_setup 主要负责文件存在性检查和提示。

    print("DeepSeek API setup function called. API key file check passed.")
    # 注意：实际的 openai.api_key 和 openai.base_url 设置可能不会在这里影响到 deepseek_completion

# 使用 DeepSeek API 生成模型响应
def deepseek_completion(
    prompt,
    model='deepseek-v3',  # 使用 DeepSeek 的模型名称
    temperature=0,
    return_response=False,
    max_tokens=500,
    retries=3,
    delay=2,
    # 可以在这里可选地传入 key_path，或者继续硬编码
    key_path='./deepseek_api_key.txt' 
):
    # 在每次调用时加载 API 密钥并创建客户端实例，这是线程安全的
    try:
        api_key = load_api_key(key_path)
    except FileNotFoundError as e:
        # deepseek_setup 应该已经检查过，但以防万一
        print(f"Error in deepseek_completion: API key file not found. {e}")
        raise ValueError("API key file not found.") from e


    # 使用显式 api_key 和 base_url 创建客户端
    # 注意：这里的 base_url 是硬编码的，覆盖了 deepseek_setup 中可能设置的全局 base_url
    client = openai.OpenAI(api_key=api_key, base_url='https://api.nuwaapi.com/v1') 

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if return_response:
                return response
            return response.choices[0].message.content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                # raise ValueError(f"All attempts failed. Last error: {e}") from e # Use 'from e' for better traceback
                raise RuntimeError(f"All attempts failed after {retries} retries. Last error: {e}") from e # RuntimeError might be more appropriate


# 在模块导入时自动执行 setup（主要用于文件存在性检查和提示）
# 注意：这将在 deepseek_utils 模块被导入时运行一次。
try:
    deepseek_setup('./deepseek_api_key.txt')
except FileNotFoundError as e:
    # 如果在导入时文件就不存在，直接退出或打印错误
    print(f"Fatal Error during module import: {e}")
    # sys.exit(1) # 在模块级别直接退出可能过于激进，让调用方处理更好
    # 只是打印错误，让后续 deepseek_completion 调用时再次抛出
    pass # Allow import to complete, the error will happen during API call

