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

def prompt_fn():
    """
    Replace fixed prompts with predefined diverse, multi-attribute prompts.
    """
    prompt = np.random.choice(prompts)
    return prompt, {}


def vqa_scorer(vqa_pipeline, local_file_path="rewards_sd1.5_vqa_program.json"):
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
    logger = Logger(log_file="training_log_vqa_program.json", image_dir="train_generated_images/sd1.5_vqa_program")
    vqa_pipeline = Pipeline(vqa_ensembler, weak_supervisor, logger)

    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save_sd1.5_vqa_program",
    }

    # 如果使用 wandb 进行日志记录，确保它已启用
    training_args.log_with = "wandb"

    # Initialize WandB
    wandb.init(project="sd1.5_vqa_program", config=script_args)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    trainer = DDPOTrainer(
        training_args,
        vqa_scorer(vqa_pipeline, local_file_path="rewards_sd1.5_vqa_program.json"),  # Pass local JSON file path
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    # Use fixed checkpoint directory and file name, so it will overwrite the latest checkpoint
    checkpoint_dir = './save_sd1.5_vqa_program/checkpoints/'
    checkpoint_file_name = get_checkpoint_file_name(checkpoint_dir)

    # Train the model
    trainer.train()

    # Save and push to hub
    trainer.save_model(checkpoint_file_name)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        