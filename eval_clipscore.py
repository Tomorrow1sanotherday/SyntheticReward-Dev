from unified_text2img.Eval_metrics import Compute_Metrics
import os
from PIL import Image
from tqdm import tqdm

metric_name = "ClipScore"
metric = Compute_Metrics(metric_name = metric_name)
# path_list = ["./generated_images/sd1.5_base_indomain", "./generated_images/sd1.5_a_indomain","./generated_images/sd1.5_vqa_indomain","./generated_images/sd1.5_v+a_indomain"]
path_list = ["./generated_images/sd1.5_v+a_mean_indomain", "./generated_images/sd1.5_v+a_mms_indomain"]
# prompt_list = [
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
#     "A gorilla sitting on a large rock surrounded by lush green vegetation.",
# ]

prompt_list = [
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

# prompt_list = [
#     "A sunflower_seed is equipped with switches the mode beige, sky blue, and dutch white izar.",
#     "There are roughrider, trefoil, fluorescent orange spatula (turner), and present townee.",
#     "There are byzantine architecture sorbian, drift_ice, and pyramidal and brown yellow color maulstick.",
#     "A quadruplicate is used for communication at the wiring and is reflected on the present vaisya.",
#     "There are common_stinkhorn, sinusoidal_projection, and dark tan, cyan azure, and academic smilo.",
#     "A churn is in the mouth of the big-boned, identical, and up forest green midland.",
#     "A bird flying over a vibrant garden full of blooming flowers.",
#     "There are globe_thistle, vivid gamboge color sliding_seat, lilliputian sunburst (design), and butterweed (wildflower).",
#     "A deer standing in a misty forest with sunlight filtering through the trees.",
#     "There is frivolous and sunset orange lidar.\n\nA ghetto_blaster is being in close proximity to the footlights.",
#     "There are dense-leaved_elodea, dark midnight blue roller_towel, and pale magenta-pink and chlorophyll green coastal_rein_orchid.",
#     "There are optician, blind_date, sailor_cap, calabar-bean_vine, and canary yellow iron_maiden.",
#     "There is undercoat (fur).\n\nA hairy and pale magenta-pink blackfoot_daisy is relied on for time by the computer_system.",
#     "There are fake angoumois_moth and servitor.\n\nA wheedler is isolating from the review.",
#     "There are smart isthmus, abator, dark pink magazine (product), and fiancee.",
#     "There are superficial and silver coyote (wolf) and ruddy pink and light hot pink louse.",
#     "There are saddle_oxford, village, minion yellow and pansy purple trailer_camp, and azerbaijani.",
#     "A cyrilla is associated with comfort the physostegia and is for the celebration of the supreme_allied_commander_atlantic;\nthe supreme_allied_commander_atlantic is wielding the physostegia.",
#     "There are very light blue female_sibling, schizocarp, international klein blue analysand, and small_stuff.",
#     "A space_writer is providing safety for the cicada_killer and is signifying the sleeper (pajama);\nthe cicada_killer is conveys affection towards the sleeper (pajama).",
# ]



# for i in path_list:
#     imgs_folder = i

#     metric.update(imgs = imgs_folder)

#     result = metric.compute()
#     print(result)

# 创建一个字典来存储图像，键是 `prompt` 序号，值是对应的图像列表
image_dict = {}
image_folder = "./generated_images/ddpo_sd1.5_b_outdomain"


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
    metric.update(prompts = [prompt]*50, images = images)
    # metric.update(prompt = prompt, images = images)
    result += metric.compute()
result = result / len(prompt_list)
print(result)


    



