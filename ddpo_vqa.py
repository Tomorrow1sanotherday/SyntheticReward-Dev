import os
import json
from dataclasses import dataclass, field
import wandb
import numpy as np
import torch
from transformers import HfArgumentParser
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from vqa_feedback import VQAEnsembler, WeakSupervisor, ImageQAModel, Pipeline
from torchvision.transforms import ToPILImage
from vqa_feedback import Logger
from util.data_process import read_only_prompts_from_jsonl


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


# Define diverse, multi-attribute prompts
prompts_file = '/home/ubuntu/dcai/SyntheticReward-Dev/asset/simple_prompts.jsonl'
prompts = read_only_prompts_from_jsonl(prompts_file)

# 全局变量跟踪当前位置
current_prompt_index = 0

def prompt_fn():
    """
    按顺序从提示列表中获取下一个prompt
    
    Returns:
        prompt: 列表中的下一个提示
        metadata: 空字典（保持与原函数兼容）
    """
    global current_prompt_index
    
    # 获取当前索引的prompt
    prompt = prompts[current_prompt_index]
    
    # 移动到下一个索引
    current_prompt_index += 1
    
    # 如果到达列表末尾，重置索引回到开始位置
    if current_prompt_index >= len(prompts):
        current_prompt_index = 0
    
    return prompt, {}


def vqa_scorer(vqa_pipeline, local_file_path="ddpo_sd1.5_vqa_simple_rewards.json"):
    """
    Use vqa_pipeline to score images based on prompts and log rewards.
    Rewards are saved both to WandB and a local JSON file.
    """
    # Ensure the local JSON file exists
    if not os.path.exists(local_file_path):
        with open(local_file_path, "w") as f:
            json.dump([], f)

    def _fn(images, prompts, metadata):
        scores = []
        all_rewards = []

         # Create a tensor-to-PIL converter
        to_pil = ToPILImage()

        for i, image in enumerate(images):
            # Check the image type
            if isinstance(image, torch.Tensor):
                # Convert tensor to PIL Image
                image = to_pil(image)
          
            # Use vqa_pipeline to calculate the score
            score = vqa_pipeline(image, prompts[i]) * 10  # Use the single prompt for scoring
            scores.append(score)

            # Reward data for the current prompt and score
            reward_data = {
                "prompt": prompts[i],
                "score": score
            }

            # Save to local JSON file
            with open(local_file_path, "r+") as f:
                try:
                    # Try to load existing data
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If the file is empty or invalid, initialize it
                    data = []

                # Append new reward data
                data.append(reward_data)

                # Write updated data back to the file
                f.seek(0)
                json.dump(data, f, indent=4)

            all_rewards.append(reward_data)

        return np.array(scores), {}

    return _fn


def image_outputs_logger(image_data, global_step, accelerate_logger):
    # For the sake of this example, we will only log the last batch of images
    # and associated data
    result = {}
    images, prompts, _, rewards, _ = image_data[-1]

    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )


def get_checkpoint_file_name(checkpoint_dir):
    """
    Use a fixed checkpoint directory and file name, so it will overwrite previous checkpoint.
    """
    return os.path.join(checkpoint_dir, 'checkpoint_latest.pth')


if __name__ == "__main__":

    # Initialize models
    vqa_model_1 = ImageQAModel(
        model_name="llavav1.6-7b", 
        torch_device="auto",
        enable_choice_search=True, 
        precision=torch.float16,
    )

    vqa_model_2 = ImageQAModel(
        model_name="llavav1.5-7b",
        torch_device="auto",
        enable_choice_search=True, 
        precision=torch.float16,
    )

    # Initialize VQA ensembler and weak supervisor
    vqa_ensembler = VQAEnsembler([vqa_model_1, vqa_model_2])
    weak_supervisor = WeakSupervisor(strategy_name="MajorityVoting")
    # Initialize Logger
    logger = Logger(log_file="ddpo_sd1.5_vqa_simple_traininglog.json", image_dir="train_generated_images/ddpo_sd1.5_vqa_simple")
    vqa_pipeline = Pipeline(vqa_ensembler, weak_supervisor, logger)

    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./ddpo_sd1.5_vqa_simple_save",
    }

    # 如果使用 wandb 进行日志记录，确保它已启用
    training_args.log_with = "wandb"

    # Initialize WandB
    wandb.init(project="ddpo_sd1.5_vqa_simple", config=script_args)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        vqa_scorer(vqa_pipeline, local_file_path="ddpo_sd1.5_vqa_simple_rewards.json"),  # Pass local JSON file path
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    # Use fixed checkpoint directory and file name, so it will overwrite the latest checkpoint
    checkpoint_dir = './ddpo_sd1.5_vqa_simple_save/checkpoints/'
    checkpoint_file_name = get_checkpoint_file_name(checkpoint_dir)

    # Train the model
    trainer.train()

 # Save the model locally using the defined checkpoint path
    trainer.save_model(checkpoint_file_name) 
    
    # Remove or comment out the code block responsible for pushing to the Hub
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name) # <- This line pushes to the Hub

    print(f"Model saved locally to {checkpoint_file_name}") # Optional: Add a confirmation message