import openai
import os
import time

# 从文件加载 API 密钥
def load_api_key(key_path='./deepseek_api_key.txt'):
    if os.path.exists(key_path):
        with open(key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    else:
        raise FileNotFoundError(f"{key_path} not found. Please ensure the API key file exists.")

# 设置 OpenAI API 密钥和基础 URL
def deepseek_setup(key_path='./deepseek_api_key.txt'):
    api_key = load_api_key(key_path)
    openai.api_key = api_key
    openai.base_url = 'https://api.nuwaapi.com/v1'  # 设置 DeepSeek 的 API 基础 URL
    print("API key has been set.")

# 使用 DeepSeek API 生成模型响应
def deepseek_completion(
    prompt,
    model='deepseek-v3',  # 使用 DeepSeek 的模型名称
    temperature=0,
    return_response=False,
    max_tokens=500,
    retries=3,
    delay=2
):
    # 创建 OpenAI 客户端实例
    key_path='./deepseek_api_key.txt'
    api_key = load_api_key(key_path)
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
                raise ValueError(f"All attempts failed. Last error: {e}")