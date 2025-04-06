import os
from PIL import Image
from evaluation import Text2ImageEvalMetric  # 请替换为你的实际模块路径
import torch


# 文件夹路径
image_folder = "generated_images/sd1.5_indomain_"  # 替换为图片文件夹的路径

# 示例生成数据
def get_gen_data_for_prompt(prompt_number):
    return {
        "prompt": f"A description for prompt{prompt_number}.",  # 使用prompt编号
    }

# 创建Text2ImageEvalMetric对象，选择需要评估的指标
metric_evaluator = Text2ImageEvalMetric(
    are_metrics_preloaded=True,  # 是否预加载所有指标
    selected_metrics=["ClipScore", "VQAScore"],  # 选择需要的指标
    device="cuda" if torch.cuda.is_available() else "cpu"  # 使用GPU还是CPU
)

# 用于存储每张图片的得分
results = {}
all_metrics_scores = {metric: [] for metric in metric_evaluator.list_metrics()}  # 初始化所有指标的评分列表

# 遍历文件夹中的图片文件
for image_name in os.listdir(image_folder):
    if image_name.endswith(".png"):
        # 从图片名提取prompt编号
        prompt_number = image_name.split("prompt")[1].split(".")[0]
        
        # 获取生成数据
        gen_data = get_gen_data_for_prompt(prompt_number)
        
        # 加载图像
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)
        
        # 评估并存储结果
        evaluation_results = metric_evaluator.eval_with_metrics(output=image, gen_data=gen_data)
        results[image_name] = evaluation_results
        
        # 累加每个指标的分数
        for metric_name, score in evaluation_results.items():
            all_metrics_scores[metric_name].append(score)

# 打印每张图片的评估结果
for image_name, evaluation in results.items():
    print(f"Results for {image_name}:")
    for metric_name, score in evaluation.items():
        print(f"  {metric_name}: {score}")

# 计算并打印每个指标的平均评分
print("\nAverage scores across all images:")
for metric_name, scores in all_metrics_scores.items():
    average_score = sum(scores) / len(scores) if scores else 0
    print(f"{metric_name}: {average_score}")
