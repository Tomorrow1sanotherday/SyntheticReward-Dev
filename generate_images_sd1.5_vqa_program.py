import torch
from diffusers import StableDiffusionPipeline
import os

# 设置模型
model_id = "runwayml/stable-diffusion-v1-5"  # 或者根据需要使用其它版本
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.load_lora_weights("save_sd1.5_vqa_program/checkpoints/checkpoint_198", weight_name="pytorch_lora_weights.safetensors", adapter_name="vqa")
# 设置显卡为第二张显卡
torch.cuda.set_device(1)  # 指定使用第二张显卡，通常cuda:1为第二张显卡

# 将模型移动到第二张显卡
pipe.to("cuda:1")

# 输入的prompt
prompts = [
    "A sunflower_seed is equipped with switches the mode beige, sky blue, and dutch white izar.",
    "There are roughrider, trefoil, fluorescent orange spatula (turner), and present townee.",
    "There are byzantine architecture sorbian, drift_ice, and pyramidal and brown yellow color maulstick.",
    "A quadruplicate is used for communication at the wiring and is reflected on the present vaisya.",
    "There are common_stinkhorn, sinusoidal_projection, and dark tan, cyan azure, and academic smilo.",
    "A churn is in the mouth of the big-boned, identical, and up forest green midland.",
    "A bird flying over a vibrant garden full of blooming flowers.",
    "There are globe_thistle, vivid gamboge color sliding_seat, lilliputian sunburst (design), and butterweed (wildflower).",
    "A deer standing in a misty forest with sunlight filtering through the trees.",
    "There is frivolous and sunset orange lidar.\n\nA ghetto_blaster is being in close proximity to the footlights.",
    "There are dense-leaved_elodea, dark midnight blue roller_towel, and pale magenta-pink and chlorophyll green coastal_rein_orchid.",
    "There are optician, blind_date, sailor_cap, calabar-bean_vine, and canary yellow iron_maiden.",
    "There is undercoat (fur).\n\nA hairy and pale magenta-pink blackfoot_daisy is relied on for time by the computer_system.",
    "There are fake angoumois_moth and servitor.\n\nA wheedler is isolating from the review.",
    "There are smart isthmus, abator, dark pink magazine (product), and fiancee.",
    "There are superficial and silver coyote (wolf) and ruddy pink and light hot pink louse.",
    "There are saddle_oxford, village, minion yellow and pansy purple trailer_camp, and azerbaijani.",
    "A cyrilla is associated with comfort the physostegia and is for the celebration of the supreme_allied_commander_atlantic;\nthe supreme_allied_commander_atlantic is wielding the physostegia.",
    "There are very light blue female_sibling, schizocarp, international klein blue analysand, and small_stuff.",
    "A space_writer is providing safety for the cicada_killer and is signifying the sleeper (pajama);\nthe cicada_killer is conveys affection towards the sleeper (pajama).",
]

# 保存图片的文件夹
output_dir = "generated_images/sd1.5_vqa_indomain_program"
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
