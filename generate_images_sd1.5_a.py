import torch
from diffusers import StableDiffusionPipeline
import os

# 设置模型
model_id = "runwayml/stable-diffusion-v1-5"  # 或者根据需要使用其它版本
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.load_lora_weights("save_model/ddpo_sd1.5_a_save/checkpoints/checkpoint_198", weight_name="pytorch_lora_weights.safetensors", adapter_name="a")
pipe.to("cuda")  # 使用GPU进行加速，如果没有GPU可以改为 'cpu'

# 输入的prompt
# prompts = [
#     "A cat sitting on a red couch with a colorful pillow next to it.",
#     "A dog running in a green field under a bright blue sky.",
#     "A white horse standing near a wooden fence in a sunny meadow.",
#     "A monkey climbing a tree with ripe bananas hanging from its branches.",
#     "A zebra drinking water from a small lake surrounded by trees.",
#     "A spider weaving an intricate web between two tall bushes.",
#     "A bird flying over a vibrant garden full of blooming flowers.",
#     "A sheep grazing in a hilly pasture with a small barn in the background.",
#     "A deer standing in a misty forest with sunlight filtering through the trees.",
#     "A cow lying on a grassy hill near a crystal-clear stream.",
#     "A lion resting under the shade of a large tree in a golden savannah.",
#     "A frog sitting on a lily pad in a quiet pond surrounded by reeds.",
#     "A chicken pecking at seeds scattered on the ground of a rustic farmyard.",
#     "A duck swimming gracefully in a calm lake during sunset.",
#     "A bee hovering near a sunflower in a bright summer field.",
#     "A hedgehog curled up on a bed of autumn leaves near a fallen log.",
#     "A kangaroo hopping across a rocky terrain under a warm orange sky.",
#     "A llama standing on a mountain trail with a panoramic view of the valley below.",
#     "A camel walking across a sandy desert with dunes in the background.",
#     "A gorilla sitting on a large rock surrounded by lush green vegetation."
# ]

prompts = [
    "A panda munching bamboo in a misty forest surrounded by tall trees.",
    "A dolphin leaping out of the sparkling ocean under a clear sky.",
    "A fox cautiously stepping through fresh snow in a quiet woodland.",
    "A squirrel gathering acorns beneath a tree with golden autumn leaves.",
    "A giraffe stretching its neck to reach leaves from a tall acacia tree.",
    "An elephant bathing in a shallow river surrounded by lush vegetation.",
    "An owl perched quietly on a branch under the silvery glow of the moon.",
    "A raccoon exploring a campsite at night illuminated by a lantern.",
    "A polar bear standing on an ice floe with snowy mountains behind.",
    "A peacock displaying its vibrant feathers in a serene palace garden.",
    "A rabbit nibbling grass beside blooming daisies in a sunny field.",
    "A koala sleeping peacefully in the branches of a eucalyptus tree.",
    "A turtle sunbathing on a rock protruding from a tranquil pond.",
    "A penguin waddling along a snowy shore with icebergs in the distance.",
    "A butterfly landing delicately on a lavender flower in a sunny meadow.",
    "A wolf howling on a rocky cliff overlooking a moonlit valley.",
    "An eagle soaring high above rugged mountains with clouds below.",
    "A crocodile basking on the muddy banks of a tropical river.",
    "A bat hanging upside-down from the ceiling of a dimly lit cave.",
    "A whale surfacing gracefully near a small boat in a deep blue ocean."
]
# 保存图片的文件夹
output_dir = "generated_images/ddpo_sd1.5_a_outdomain"
os.makedirs(output_dir, exist_ok=True)

# 生成50张图片，每个prompt对应不同的种子
for idx, prompt in enumerate(prompts):
    print(f"Generating images for prompt: {prompt}")
    index = 0

    # 生成50张图
    for seed in range(50):
        # 设置种子
        generator = torch.manual_seed(seed)
        
        # 生成图片
        image = pipe(prompt, guidance_scale=5.0, num_inference_steps=50, generator=generator).images[0]
        
        # 保存图片
        image.save(os.path.join(output_dir, f"image_prompt{idx}_{index}.png"))
        print(f"Image {index} saved for prompt {idx}")
        index+=1

print("All images generated and saved.")
