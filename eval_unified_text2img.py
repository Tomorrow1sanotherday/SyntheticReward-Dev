from unified_text2img.Eval_metrics import Compute_Metrics
import os
from PIL import Image
from tqdm import tqdm

metric_name = "HPSv2"
metric = Compute_Metrics(metric_name = metric_name)
path_list = ["./generated_images/sd1.5_base_indomain", "./generated_images/sd1.5_a_indomain","./generated_images/sd1.5_vqa_indomain","./generated_images/sd1.5_v+a_indomain"]
prompt_list = [
    "A cat sitting on a red couch with a colorful pillow next to it.",
    "A dog running in a green field under a bright blue sky.",
    "A white horse standing near a wooden fence in a sunny meadow.",
    "A monkey climbing a tree with ripe bananas hanging from its branches.",
    "A zebra drinking water from a small lake surrounded by trees.",
    "A spider weaving an intricate web between two tall bushes.",
    "A bird flying over a vibrant garden full of blooming flowers.",
    "A sheep grazing in a hilly pasture with a small barn in the background.",
    "A deer standing in a misty forest with sunlight filtering through the trees.",
    "A cow lying on a grassy hill near a crystal-clear stream.",
    "A lion resting under the shade of a large tree in a golden savannah.",
    "A frog sitting on a lily pad in a quiet pond surrounded by reeds.",
    "A chicken pecking at seeds scattered on the ground of a rustic farmyard.",
    "A duck swimming gracefully in a calm lake during sunset.",
    "A bee hovering near a sunflower in a bright summer field.",
    "A hedgehog curled up on a bed of autumn leaves near a fallen log.",
    "A kangaroo hopping across a rocky terrain under a warm orange sky.",
    "A llama standing on a mountain trail with a panoramic view of the valley below.",
    "A camel walking across a sandy desert with dunes in the background.",
    "A gorilla sitting on a large rock surrounded by lush green vegetation.",
]



# for i in path_list:
#     imgs_folder = i

#     metric.update(imgs = imgs_folder)

#     result = metric.compute()
#     print(result)

# 创建一个字典来存储图像，键是 `prompt` 序号，值是对应的图像列表
image_dict = {}
image_folder = "./generated_images/sd1.5_base_indomain"


# 遍历文件夹中的文件
for image_file in os.listdir(image_folder):
    # 提取文件名，例如 'image_prompt0_0.png' 中的 '0' 表示 `prompt 0`
    if image_file.endswith(".png"):
        # 分割字符串，获取 'prompt0' 和 '0.png'
        parts = image_file.split('_')

        # 提取 'prompt0' 中的数字部分（这里的数字是 0）
        prompt_index = int(parts[1].replace('prompt', ''))  # 去除 'prompt' 后转换为整数
        
        # 加载图像
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        
        # 如果该 `prompt` 序号的列表不存在，则初始化一个空列表
        if prompt_index not in image_dict:
            image_dict[prompt_index] = []
        # image = image_path
        
        # 将图像添加到对应的列表中
        image_dict[prompt_index].append(image)

result = 0
for prompt_index, images in tqdm(image_dict.items(), desc="Processing Prompts"):
    prompt = prompt_list[prompt_index]
    # metric.update(prompts = [prompt]*50, images = images)
    metric.update(prompt = prompt, images = images)
    print(result)
    print(metric.compute())
result = result / len(prompt_list)
print(result)


    



