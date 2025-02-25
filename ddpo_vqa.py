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


def vqa_scorer(vqa_pipeline, local_file_path="rewards_sd1.5_vqa.json_2"):
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
            score = vqa_pipeline(image, prompts[i])  # Use the single prompt for scoring
            scores.append(score)

            # Reward data for the current prompt and score
            reward_data = {
                "prompt": prompts[i],
                "score": score,
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
    logger = Logger(log_file="training_log_vqa_2.json", image_dir="train_generated_images/sd1.5_vqa_2")
    vqa_pipeline = Pipeline(vqa_ensembler, weak_supervisor, logger)

    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save_sd1.5_vqa_2",
    }

    # 如果使用 wandb 进行日志记录，确保它已启用
    training_args.log_with = "wandb"

    # Initialize WandB
    wandb.init(project="sd1.5_vqa_2", config=script_args)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        vqa_scorer(vqa_pipeline, local_file_path="rewards_sd1.5_vqa_2.json"),  # Pass local JSON file path
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    # Use fixed checkpoint directory and file name, so it will overwrite the latest checkpoint
    checkpoint_dir = './save_sd1.5_vqa_2/checkpoints/'
    checkpoint_file_name = get_checkpoint_file_name(checkpoint_dir)

    # Train the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(checkpoint_file_name)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        