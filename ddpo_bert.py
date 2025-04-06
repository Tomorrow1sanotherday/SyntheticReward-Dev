import os
import json
import torch
from dataclasses import dataclass, field
import wandb
import numpy as np
from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from vqa_feedback import VQAEnsembler, WeakSupervisor, ImageQAModel, Pipeline
from torchvision.transforms import ToPILImage
from bert_score import score  # Import BERTScore for similarity-based rewards
from vqa_feedback import Logger
PROMPT = "Answer concisely: what is going on in this image?."

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

    # Initialize models
llava_model = ImageQAModel(
    model_name="llavav1.5-7b", 
    torch_device="auto",
    enable_choice_search=True, 
    precision=torch.float16,
)

prompts = [
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
    """
    Replace fixed prompts with predefined diverse, multi-attribute prompts.
    """
    prompt = np.random.choice(prompts)
    return prompt, {}

def bertscore_scorer(local_file_path="rewards_sd1.5_bertscore_4.json"):
    """
    Use BERTScore to calculate similarity-based rewards between generated and target descriptions.
    """
    if not os.path.exists(local_file_path):
        with open(local_file_path, "w") as f:
            json.dump([], f)

    def _fn(images, prompts, metadata):
        all_rewards = []
        to_pil = ToPILImage()
        save_dir = './train_generated_images/ddpo_sd1.5_b'

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 获取已保存的图像文件数量，确保命名不重复
        existing_files = os.listdir(save_dir)
        existing_images = [f for f in existing_files if f.startswith("image_") and f.endswith(".png")]
        start_index = len(existing_images) + 1  # 从下一个可用的索引开始

        for i, image in enumerate(images):
            # 生成描述
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image
                image = to_pil(image)
            generated_description = llava_model.qa(image, PROMPT)
            print(generated_description)

            # 计算 BERTScore 来获取奖励分数
            _, _, bert_scores = score([generated_description], [prompts[i]], lang="en", verbose=True)
            reward_score = bert_scores.mean().item()*10  # Use average BERTScore as reward

            # 使用递增的索引命名文件，确保不重复
            image_filename = f"{save_dir}/image_{start_index}.png"
            image.save(image_filename)
            print(f"已保存图像: {image_filename}")

            # 将图像路径、描述和奖励保存到 JSON 文件
            reward_data = {
                "prompt": prompts[i],
                "generated_description": generated_description,
                "bert_score": reward_score,
                "image_path": image_filename  # 保存图像的路径
            }

            # 保存到 JSON 文件
            with open(local_file_path, "r+") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []

                data.append(reward_data)
                f.seek(0)
                json.dump(data, f, indent=4)

            all_rewards.append(reward_score)
            start_index += 1  # 增加索引

        return np.array(all_rewards), {}

    return _fn

def image_outputs_logger(image_data, global_step, accelerate_logger):
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(result, step=global_step)

def get_checkpoint_file_name(checkpoint_dir):
    return os.path.join(checkpoint_dir, 'checkpoint_latest.pth')

if __name__ == "__main__":

    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./ddpo_sd1.5_b_save",
    }

    training_args.log_with = "wandb"
    wandb.init(project="ddpo_sd1.5_b", config=script_args)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        bertscore_scorer(local_file_path="ddpo_sd1.5_b_rewards.json"),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    checkpoint_dir = './ddpo_sd1.5_b_save/checkpoints/'
    checkpoint_file_name = get_checkpoint_file_name(checkpoint_dir)

    # Train the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(checkpoint_file_name)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
