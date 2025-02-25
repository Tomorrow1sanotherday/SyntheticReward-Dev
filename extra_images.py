import json
import shutil
import os

# 定义文件路径
json_file_path = 'training_log_v+a.json'  # 你的json文件路径
source_folder = 'train_generated_images/sd1.5_v+a/'  # 图片所在的文件夹路径
destination_folder = 'filtered_images/cat_v+a/'  # 目标文件夹路径

# 具体的 prompt 内容
target_prompt = "A cat sitting on a red couch with a colorful pillow next to it."

# 如果目标文件夹不存在，创建它
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 打开并加载JSON数据
with open(json_file_path, 'r') as f:
    data = json.load(f)

# 遍历JSON数据并复制符合特定prompt的图片
for entry in data:
    if entry['prompt'] == target_prompt:  # 检查prompt是否与目标完全匹配
        image_path = entry['image_path']
        
        # 确保源文件路径拼接正确
        destination_image = os.path.join(destination_folder, os.path.basename(image_path))  # 目标文件路径
        
        # 复制图片
        try:
            shutil.copy(image_path, destination_image)
            print(f'Copied {image_path} to {destination_image}')
        except FileNotFoundError:
            print(f"Error: {image_path} does not exist.")
            continue

print("Finished copying all matching 'cat' related images.")
