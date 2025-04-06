from unified_text2img.Eval_metrics import Compute_Metrics
import os
from PIL import Image
from tqdm import tqdm

metric_name = "InceptionScore"
metric = Compute_Metrics(metric_name = metric_name)
# path_list = ["./generated_images/sd1.5_base_indomain_program", "./generated_images/sd1.5_a_indomain_program", "./generated_images/sd1.5_vqa_indomain_program", "./generated_images/sd1.5_v+a_indomain_program"]
# path_list = ["./generated_images/sd1.5_base_outdomain_program", "./generated_images/sd1.5_a_outdomain_program", "./generated_images/sd1.5_vqa_outdomain_program", "./generated_images/sd1.5_v+a_outdomain_program"]
# path_list = ["./generated_images/ddpo_sd1.5_base_outdomain", "./generated_images/ddpo_sd1.5_a_outdomain",
#              "./generated_images/ddpo_sd1.5_vqa_outdomain", "./generated_images/ddpo_sd1.5_v+a_mean_outdomain",
#              "./generated_images/ddpo_sd1.5_v+a_mms_outdomain","./generated_images/ddpo_sd1.5_b_outdomain",
#              "./generated_images/ddpo_sd1.5_v+b_mean_outdomain", "./generated_images/ddpo_sd1.5_v+b_mms_outdomain"]
path_list = ["./generated_images/ddpo_sd1.5_base_indomain", "./generated_images/ddpo_sd1.5_a_indomain",
             "./generated_images/ddpo_sd1.5_vqa_indomain", "./generated_images/ddpo_sd1.5_v+a_mean_indomain",
             "./generated_images/ddpo_sd1.5_v+a_mms_indomain", "./generated_images/ddpo_sd1.5_b_indomain",
             "./generated_images/ddpo_sd1.5_v+b_mean_indomain", "./generated_images/ddpo_sd1.5_v+b_mms_indomain"]






for i in path_list:
    imgs_folder = i

    metric.update(imgs = imgs_folder)

    result = metric.compute()
    print(result)



    



