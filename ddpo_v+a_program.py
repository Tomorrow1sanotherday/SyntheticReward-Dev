import os
import json
from dataclasses import dataclass, field
import wandb
import numpy as np
import torch
import torch.nn as nn
from transformers import HfArgumentParser, CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from trl import DDPOConfig, DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from vqa_feedback import VQAEnsembler, WeakSupervisor, ImageQAModel, Pipeline
from torchvision.transforms import ToPILImage
from vqa_feedback import Logger
from vqa_feedback import RewardEnsembler

# 添加美学评分相关的类
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
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)

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

# 保持原有的prompts列表不变
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
    prompt = np.random.choice(prompts)
    return prompt, {}

def combined_scorer(vqa_pipeline, aesthetic_scorer, local_file_path="rewards_sd1.5_v+a_mms_program.json"):
    """
    结合VQA和美学评分的评分器
    """
    def _fn(images, prompts, metadata):
        scores = []
        all_rewards = []
        to_pil = ToPILImage()
        reward_ensembler = RewardEnsembler()

        # 获取美学评分
        images_tensor = torch.stack([img if isinstance(img, torch.Tensor) else torch.from_numpy(img) for img in images])
        aesthetic_scores = aesthetic_scorer(images_tensor)

        for i, image in enumerate(images):
            if isinstance(image, torch.Tensor):
                image = to_pil(image)
            
            # 获取VQA评分并调整范围到1-10
            vqa_score = vqa_pipeline(image, prompts[i])
            rewards = [vqa_score, aesthetic_scores[i].item()]
            
            # 结合两个评分
            combined_score = reward_ensembler.normalize(rewards = rewards, method = 'weight', weights = [10, 1])
            combined_score = reward_ensembler.aggregate_rewards(combined_score, method="mean_minus_std")
            scores.append(combined_score)

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
                "vqa_score": vqa_score,
                "aesthetic_score": aesthetic_scores[i].item(),
                "combined_score": combined_score
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

    return _fn


def image_outputs_logger(image_data, global_step, accelerate_logger):
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
    return os.path.join(checkpoint_dir, 'checkpoint_latest.pth')

if __name__ == "__main__":
    # 初始化VQA模型
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

    # 初始化VQA组件
    vqa_ensembler = VQAEnsembler([vqa_model_1, vqa_model_2])
    weak_supervisor = WeakSupervisor(strategy_name="MajorityVoting")
    logger = Logger(log_file="training_log_v+a_mms_program.json", image_dir="train_generated_images/sd1.5_v+a_mms_program")
    vqa_pipeline = Pipeline(vqa_ensembler, weak_supervisor, logger)

    # 初始化美学评分器
    aesthetic_model = AestheticScorer(
        model_id="trl-lib/ddpo-aesthetic-predictor",
        model_filename="aesthetic-model.pth",
        dtype=torch.float32,
    )
    aesthetic_model = aesthetic_model.to(torch.device('cpu'))


    parser = HfArgumentParser((ScriptArguments, DDPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    training_args.project_kwargs = {
        "logging_dir": "./logs",
        "automatic_checkpoint_naming": True,
        "total_limit": 5,
        "project_dir": "./save_sd1.5_v+a_mms_program",
    }

    training_args.log_with = "wandb"
    wandb.init(project="sd1.5_v+a_mms_program", config=script_args)

    pipeline = DefaultDDPOStableDiffusionPipeline(
        script_args.pretrained_model,
        pretrained_model_revision=script_args.pretrained_revision,
        use_lora=script_args.use_lora,
    )

    # 使用组合评分器
    trainer = DDPOTrainer(
        training_args,
        combined_scorer(vqa_pipeline, aesthetic_model, local_file_path="rewards_sd1.5_v+a_mms_program.json"),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    checkpoint_dir = './save_sd1.5_v+a_mms_program/checkpoints/'
    checkpoint_file_name = get_checkpoint_file_name(checkpoint_dir)

    trainer.train()
    trainer.save_model(checkpoint_file_name)
    
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
