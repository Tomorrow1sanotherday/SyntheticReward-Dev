cd SyntheticReward-Dev
1、pip install -r pip_requirements.txt
2、conda install --file conda_requirements.txt -c conda-forge
修改deepseek_api_key.txt，改成自己的api
3、HF_ENDPOINT=https://hf-mirror.com python ddpo_vqa_simple.py     --num_epochs=210     --train_gradient_accumulation_steps=1     --sample_num_steps=50     --sample_batch_size=6     --train_batch_size=3     --sample_num_batches_per_epoch=4     --per_prompt_stat_tracking=True     --per_prompt_stat_tracking_buffer_size=32     --tracker_project_name="stable_diffusion_training"     --log_with="wandb"