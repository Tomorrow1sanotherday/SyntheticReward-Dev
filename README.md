# 训练代码

1. 进入 `SyntheticReward-Dev` 目录：
    ```bash
    cd SyntheticReward-Dev
    ```

2. 安装 Python 依赖：
    ```bash
    pip install -r pip_requirements.txt
    ```

3. 安装 Conda 依赖：
    ```bash
    conda install --file conda_requirements.txt -c conda-forge
    ```

4. 修改 `deepseek_api_key.txt` 文件，替换为自己的 API 密钥。

5. 运行训练代码：
    ```bash
    HF_ENDPOINT=https://hf-mirror.com python ddpo_vqa_simple.py \
      --num_epochs=210 \
      --train_gradient_accumulation_steps=1 \
      --sample_num_steps=50 \
      --sample_batch_size=6 \
      --train_batch_size=3 \
      --sample_num_batches_per_epoch=4 \
      --per_prompt_stat_tracking=True \
      --per_prompt_stat_tracking_buffer_size=32 \
      --tracker_project_name="stable_diffusion_training" \
      --log_with="wandb"
    ```

# 生成 `qid_data.json` 文件

1. 进入 `SyntheticReward-Dev` 目录并创建 `qid_data.json` 文件：
    ```bash
    cd SyntheticReward-Dev
    ```

2. 修改 `deepseek_api_key.txt` 文件，替换为自己的 API 密钥。

3. 运行生成 `qid_data.json` 文件的代码：
    ```bash
    python get_qid_data.py
    ```
