from PIL import Image
import io
import numpy as np
import torch
import os
from torchvision.transforms import ToPILImage
import json

def light_reward():
    def _fn(images, prompts, metadata):
        reward = images.reshape(images.shape[0],-1).mean(1)
        return np.array(reward.cpu().detach()),{}
    return _fn


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


def aesthetic_score():
    from d3po_pytorch.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

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



