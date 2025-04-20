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
import shutil


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
global_epoch = -2

# 获取当前脚本文件的绝对路径
script_path = os.path.abspath(__file__)

# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(script_path)

def prompt_fn():
    global current_prompt_index
    prompt = prompts[current_prompt_index]
    current_prompt_index = (current_prompt_index + 1) % len(prompts)
    return prompt, {}


def vqa_scorer(vqa_pipeline, local_file_path="ddpo_sd1.5_vqa_simple_rewards.json"):
    if not os.path.exists(local_file_path):
        with open(local_file_path, "w") as f:
            json.dump([], f)

    def _fn(images, prompts, metadata):
        scores = []
        all_rewards = []
        to_pil = ToPILImage()

        for i, image in enumerate(images):
            if isinstance(image, torch.Tensor):
                image = to_pil(image)
            score = vqa_pipeline(image, prompts[i]) * 10
            scores.append(score)
            reward_data = {"prompt": prompts[i], "score": score}
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


# 全局维护 top5 reward mean 列表及对应 checkpoint 名称
best_means = []  # 存 float
best_ckpts = []  # 存 checkpoint 子目录名，比如 'checkpoint-1000'



def image_outputs_logger(image_data, global_step, accelerate_logger):
    """
    在原有日志图片的基础上，新增 top5 reward_mean checkpoint 复制逻辑。
    """
    global best_means, best_ckpts, global_epoch

    # 取最后一批
    images, prompts, _, rewards, _ = image_data[-1]
    # 计算本批次 mean reward
    mean_reward = rewards.float().mean().item()
    print(rewards)
    print(mean_reward)

    # 如果还没满5个，或者 mean_reward 大于当前最小的 top5
    if len(best_means) < 5 or mean_reward > min(best_means):
        # 确定要插入的位置
        if mean_reward in best_means:
            # 如果正好重复均值，则跳过复制
            pass
        else:
            # 如果已满5个，先移除最小的
            if len(best_means) >= 5:
                idx_min = best_means.index(min(best_means))
                best_means.pop(idx_min)
                old_ckpt = best_ckpts.pop(idx_min)
                # 可选：删除对应旧文件夹
                old_path = os.path.join(best_ckpt_target_dir, old_ckpt)
                if os.path.exists(old_path):
                    shutil.rmtree(old_path)
            # 插入新的
            best_means.append(mean_reward)
            # 假设当前最新 checkpoint 子目录是 'checkpoint-{global_step}'
            ckpt_name = f'checkpoint_{global_epoch}'
            best_ckpts.append(ckpt_name)

            # 复制 checkpoint
            src_dir = os.path.join(checkpoint_dir, ckpt_name)
            dst_dir = os.path.join(best_ckpt_target_dir, ckpt_name)
            if os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
                # 复制整个目录
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print(f"已保存新的 top5 checkpoint: {ckpt_name}, mean_reward={mean_reward:.4f}")
            else:
                print(src_dir)
                print(dst_dir)

    # 否则不做任何操作

    # 原有的图片日志逻辑
    result = {}
    for i, image in enumerate(images):
        prompt = prompts[i]
        reward = rewards[i].item()
        result[f"{prompt:.25} | {reward:.2f}"] = image.unsqueeze(0).float()

    accelerate_logger.log_images(
        result,
        step=global_step,
    )
    global_epoch += 1





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
        "total_limit": None,
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

    # checkpoint 目录，用于复制
    checkpoint_dir = 'ddpo_sd1.5_vqa_simple_save/checkpoints'
    # 你要复制到的目标文件夹
    best_ckpt_target_dir = 'ddpo_sd1.5_vqa_simple_save/best_checkpoints'

    trainer = DDPOTrainer(
        training_args,
        vqa_scorer(vqa_pipeline, local_file_path="ddpo_sd1.5_vqa_simple_rewards.json"),
        prompt_fn,
        pipeline,
        image_samples_hook=image_outputs_logger,
    )

    trainer.train()
    print("训练结束")

