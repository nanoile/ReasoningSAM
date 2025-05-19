import sys
import os
import json
from instruct_sam.matching import init_clip_model
from instruct_sam.metrics import get_cat_map, evaluate_ap_f1, get_all_counts
from instruct_sam.counting import eval_detection_counting_coco
from instruct_sam.instruct_sam import InstructSAM
from tqdm import tqdm
import time
from instruct_sam.metrics import convert_coco_raw_to_coco_pred, create_completion


device = "cuda:6"
print(device)

task = 'oed'    # oed, osd

dataset = 'nwpu'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/nwpu/osd/coco_preds'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/owlv2/gpt-4o-2024-11-20_transports_owl-b_nwpu_raw.json'
coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/GeoPixel/vis2/geo_pixel_coco.json'
# coco_results_raw_path = None
img_dir = '/home/zkcs/datasets/nwpu/pos_img'
ann_path = '/home/zkcs/datasets/nwpu/nwpu_ins_coco.json'
count_dir = '/home/zkcs/zyj/label-engine/object_counts/nwpu/qwen7b_transports_visible'
rp_path = './region_proposals/nwpu/sam2-l.json'
unseen_classes = ['airplane', 'ship', 'vehicle'] if 'transports' in coco_results_raw_path else ['baseball_field', 'tennis_court', 'basketball_court', 'track_field']

# dataset = 'dior'
# # coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/dior/osd/coco_preds/gpt-4o-2024-11-20_transports_visible_sam2-con0.75_georsclip_raw.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/owlv2/gpt-4o-2024-11-20_sports_visible_owl-b_oed_dior_raw.json'
# # coco_results_raw_path = None
# img_dir = '/home/zkcs/datasets/dior/JPEGImages-trainval'
# ann_path = '/home/zkcs/datasets/dior/dior_ins_val.json'
# count_dir = './object_counts/dior/qwen7b_osd_sports_visible'
# rp_path = '/home/zkcs/zyj/label-engine/region_proposals/dior/sam2-l_RLE_val.json'
# unseen_classes = ['airplane', 'ship', 'vehicle'] if 'transports' in coco_results_raw_path else ['baseball field', 'basketball court', 'golf field', 'ground track field', 'tennis court', 'stadium']

confidence = 0.75
min_crop_width = 0
match_model = 'skyclip'
use_vocab = False
LLM_judge = False
clip_thr = 0.95
det_threshold = 0.01

map_model = 'georsclip'

with open(ann_path, 'r') as f:
    val_anns = json.load(f)

if coco_results_raw_path is None:
    count = count_dir.split('/')[-1]

    save_preds_path = f'./results/{dataset}/{task}/coco_preds/{count}_sam2_{match_model}_raw_geo_pixel.json'
    print(f'Save {task} predictions to {save_preds_path}')
    
    model, tokenizer, preprocess = init_clip_model(match_model, device)

    with open(rp_path, 'r') as f:
        rp_preds = json.load(f)
    img_name_list = [img['file_name'] for img in val_anns['images']]

    predictor = InstructSAM(val_anns, img_dir, count_dir, rp_preds)
    predictor.calculate_vocab_text_features(model, tokenizer)

    coco_results_raw = []

    # Speed metrics
    process_time = {}

    for image_name in tqdm(img_name_list):
        image_path = os.path.join(img_dir, image_name)
        predictor.set_image(image_path)
        predictor.load_rps_and_cnts(thr=confidence)
        start_time = time.time()
        predictor.calculate_pred_text_features(model, tokenizer, use_vocab=use_vocab)
        predictor.match_boxes_and_labels(model, preprocess, min_crop_width=min_crop_width, zero_count_warning=False)

        # record process time
        process_time[image_name] = {'time': time.time() - start_time}
        
        for label, bbox, segmentation, score in zip(predictor.labels_final, predictor.boxes_final, predictor.segmentations_final, predictor.scores_final):
            coco_results_raw.append({
                "image_id": predictor.img_id,
                "bbox": bbox,
                "score": score,
                "segmentation": segmentation,
                "label": label
            })
            
    with open(save_preds_path, 'w') as f:
        json.dump(coco_results_raw, f)

else:
    with open(coco_results_raw_path, 'r') as f:
        coco_results_raw = json.load(f)

predictor = InstructSAM(val_anns)

# 计算 all_counts
all_counts = get_all_counts(coco_results_raw)

if LLM_judge:
    pred_cat_str = '"' + '", "'.join(list(all_counts.keys())) + '"'
    dataset_cat_str = '"' + '", "'.join(predictor.categories) + '"'

    prompt = f"""# Task
    You are tasked with mapping predicted categories from a GPT counter to a defined set of dataset categories. The GPT counter has predicted the following categories:
    {pred_cat_str}
    The defined dataset categories are:
    {dataset_cat_str}
    # Rules for mapping
    1. Synonyms and Related Terms:
        - Any predicted category that is a synonym a dataset category should be mapped. For example:
        - "boat" should be mapped to "ship".
        - "track" and "playground" should be mapped to "ground track field".
    2. Remote Sensing Context:
        - The images are in satellite view, so the exact meaning of categories should align with the remote sensing domain. For example: "tank" and "container" should be mapped to "storage tank".

    3. Subcategories to Parent Categories
        - Subcategories should be mapped to their parent categories. For example: "van" and "construction_vehicle" should be mapped to "vehicle".

    # Output

    Provide the mapping in JSON format where keys are the predicted categories and values are the mapped dataset categories. Only include predicted categories that can be mapped based on the rules above."""

    # deepseek-chat, gemini-2.0-flash-001 'chatgpt-4o-latest
    pred2cat = create_completion(prompt, model_name='deepseek-chat', json_output=True)   #
else:
    model, tokenizer, preprocess = init_clip_model(map_model, device)
    pred2cat = get_cat_map(model, tokenizer, predictor.categories, all_counts, threshold=clip_thr)

# 打印结果以验证
print(f"Total unique classes: {len(all_counts)}")
print(f"Examples of counts: {list(all_counts.items())[:5]}")  # 显示前5个类别的计数

coco_predictions = convert_coco_raw_to_coco_pred(coco_results_raw, pred2cat, val_anns, det_threshold=0.01)
print(f"Total {len(coco_results_raw)} raw predictions, {len(coco_predictions)} predictions after mapping\n")

from instruct_sam.metrics import calculate_f1_scores
import copy
import numpy as np
# Thresholds to evaluate
thresholds = [round(x, 2) for x in list(np.arange(0.01, 0.5, 0.01))]
# thresholds = [round(x, 4) for x in list(np.arange(0.15, 0.3, 0.0025))]
mf1_values = []

# Evaluate mAP50 for each threshold
coco_predictions_copy = copy.deepcopy(coco_predictions)
for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
    coco_predictions_filtered = convert_coco_raw_to_coco_pred(coco_results_raw, pred2cat, val_anns, det_threshold=threshold)
    # print(len(coco_predictions_filtered))
    # Calculate mF1
    class_metrics, mean_f1_score, _ = calculate_f1_scores(val_anns, coco_predictions_filtered)
    mf1_values.append(mean_f1_score)
    print(f'threshold: {threshold}, mF1: {mean_f1_score}')

# calculate the best threshold
best_threshold = thresholds[np.argmax(mf1_values)]
print(f'best threshold: {best_threshold}, mF1: {mf1_values[np.argmax(mf1_values)]}')
coco_predictions = convert_coco_raw_to_coco_pred(coco_results_raw, pred2cat, val_anns, det_threshold=best_threshold)
print('## Evaluating box AP and F1 ##')
box_metrics = evaluate_ap_f1(val_anns, coco_predictions, iouType='bbox', unseen=unseen_classes, save_path=None, print_result=False)
print(f"mF1: {box_metrics['mF1']:.1f}, mF1_unseen: {box_metrics['mF1_unseen']:.1f}")

print('\n\n## Evaluating mask AP and F1 ##')
mask_metrics = evaluate_ap_f1(val_anns, coco_predictions, iouType='segm', unseen=unseen_classes, save_path=None, print_result=False)
print(f"mF1: {mask_metrics['mF1']:.1f}, mF1_unseen: {mask_metrics['mF1_unseen']:.1f}")

print('## Evaluating counting metrics ##')
cnt_metrics = eval_detection_counting_coco(coco_predictions, val_anns, unseen=unseen_classes, print_result=False)
print(f'Cnt mean metrics: {cnt_metrics["macro_avg"]}')
print(f'Cnt unseen metrics: {cnt_metrics["unseen_avg"]}')