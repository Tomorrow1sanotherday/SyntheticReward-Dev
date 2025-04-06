import os
import shutil
import json
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from transformers import CLIPModel, CLIPProcessor, HfArgumentParser, is_torch_npu_available, is_torch_xpu_available

from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
import wandb
from torchvision.transforms import ToPILImage


@dataclass
class ScriptArguments:
    pretrained_model: str = field(
        default="runwayml/stable-diffusion-v1-5", metadata={"help": "the pretrained model to use"}
    )
    pretrained_revision: str = field(default="main", metadata={"help": "the pretrained model revision to use"})
    hf_hub_model_id: str = field(
        default="ddpo-finetuned-stable-diffusion", metadata={"help": "HuggingFace repo to save model weights to"}
    )
    hf_hub_aesthetic_model_id: str = field(
        default="trl-lib/ddpo-aesthetic-predictor",
        metadata={"help": "HuggingFace model ID for aesthetic scorer model weights"},
    )
    hf_hub_aesthetic_model_filename: str = field(
        default="aesthetic-model.pth",
        metadata={"help": "HuggingFace model filename for aesthetic scorer model weights"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, *, dtype, model_id, model_filename):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        try:
            cached_path = hf_hub_download(model_id, model_filename)
        except EntryNotFoundError:
            cached_path = os.path.join(model_id, model_filename)
        state_dict = torch.load(cached_path, map_location=torch.device("cpu"), weights_only=True)
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip.get_image_features(**inputs)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def aesthetic_scorer(hub_model_id, model_filename):
    scorer = AestheticScorer(
        model_id=hub_model_id,
        model_filename=model_filename,
        dtype=torch.float32,
    )
    if is_torch_npu_available():
        scorer = scorer.npu()
    elif is_torch_xpu_available():
        scorer = scorer.xpu()
    else:
        scorer = scorer.cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


# list of example prompts to feed stable diffusion
animals = [
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


def prompt_fn():
    return np.random.choice(animals), {}


def save_images_and_rewards(images, aesthetic_scores, prompts, local_file_path="ddpo_sd1.5_a_rewards.json"):
    """
    保存生成的图像和对应的美学评分
    """
    to_pil = ToPILImage()
    scores = []
    all_rewards = []
    
    # 创建保存图像的文件夹
    if not os.path.exists('train_generated_images/ddpo_sd1.5_a'):
        os.makedirs('train_generated_images/ddpo_sd1.5_a')

    # 计算当前文件夹已有的图像数量，防止文件重名
    existing_images = len([f for f in os.listdir('train_generated_images/ddpo_sd1.5_a') if f.endswith('.png')])
    
    for i, image in enumerate(images):
        if isinstance(image, torch.Tensor):
            image = to_pil(image)
        
        # 确保图像命名唯一，避免覆盖现有图像
        save_image_path = f"train_generated_images/ddpo_sd1.5_a/image_{existing_images + i + 1}.png"
        image.save(save_image_path)
        
        aesthetic_score = aesthetic_scores[i].item()
        scores.append(aesthetic_score)

        # 读取现有数据以确定下一个ID
        if os.path.exists(local_file_path):
            with open(local_file_path, "r+") as f:
                try:
                    existing_data = json.load(f)
                    next_id = max([item.get("id", 0) for item in existing_data], default=0) + 1
                except json.JSONDecodeError:
                    next_id = 1
        else:
            # Create the file if it doesn't exist
            with open(local_file_path, "w") as f:
                json.dump([], f)  # Initialize the file with an empty list
            next_id = 1

        reward_data = {
            "id": next_id,
            "prompt": prompts[i],
            "aesthetic_score": aesthetic_score,
            "image_path": save_image_path  # 保存图片路径
        }

        # 保存到本地JSON
        with open(local_file_path, "r+") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(reward_data)
            f.seek(0)
            json.dump(data, f, indent=4)

        all_rewards.append(reward_data)

    return np.array(scores), {}


def image_outputs_logger(image_data, global_step, accelerate_logger):
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    # 调用保存图片和美学评分的函数
    save_images_and_rewards(images, rewards, prompts, local_file_path="ddpo_sd1.5_a_rewards.json")

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./ddpo_sd1.5_a_save",
    }

    training_args.log_with = "wandb"
    wandb.init(project="ddpo_sd1.5_a", config=script_args)

    # 删除现有的检查点目录并重新创建
    checkpoint_dir = './ddpo_sd1.5_a_save/checkpoints/'
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)  # 删除现有检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)  # 创建新的检查点目录

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    # 使用美学评分器
    trainer = DDPOTrainer(
        training_args,
        aesthetic_scorer(script_args.hf_hub_aesthetic_model_id, script_args.hf_hub_aesthetic_model_filename),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()

    # Save and push to hub
    print('*' * 100)
    trainer.save_model(training_args.output_dir)
    print('*' * 100)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
