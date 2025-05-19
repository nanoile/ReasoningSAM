import sys
sys.path.append('/home/zkcs/zyj/label-engine')
import os
import json
from instruct_sam.matching import init_clip_model
from instruct_sam.metrics import convert_predictions_to_coco, evaluate_ap_f1, evaluate_mIoU
from instruct_sam.instruct_sam import InstructSAM
from tqdm import tqdm
import numpy as np

device = "cuda:1"

# dataset = 'nwpu'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/GroundingDINO/GDINO_ovd_nwpu_coco_update.json'
# save_results = False
# # coco_results_raw_path = None    
# img_dir = '/home/zkcs/datasets/nwpu/pos_img'
# ann_path = '/home/zkcs/datasets/nwpu/nwpu_ins_coco.json' 
# count_dir = '/home/zkcs/zyj/label-engine/object_counts/nwpu/gpt-4o-2024-11-20_ov_align'
# rp_path = './region_proposals/nwpu/sam2-s.json'
# unseen_classes = ['ship', 'basketball_court', 'harbor']

# dataset = 'dior'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/dior/ovd/coco_preds/owl-b_ovd_dior_coco.json'
# save_results = False
# # coco_results_raw_path = None
# img_dir = '/home/zkcs/datasets/dior/JPEGImages-test'
# ann_path = '/home/zkcs/datasets/dior/dior_ins_test.json'
# count_dir = '/home/zkcs/zyj/label-engine/object_counts/dior/qwen7b_ovd_align_testset'
# rp_path = './region_proposals/dior/sam2-l_test.json'
# unseen_classes = ['airport', 'basketball court', 'ground track field', 'windmill']

dataset = 'xBD'
img_dir = '/home/zkcs/datasets/xBD_raw/test/images/pre-event'
ann_path = '/home/zkcs/datasets/xBD_raw/test/xBD_building_location.json'
count_dir = '/home/zkcs/zyj/label-engine/object_counts/xBD_building/gpt-4o-2024-11-20_finetune_v2'
# rp_path = '/home/zkcs/zyj/label-engine/region_proposals/xBD/xBD_building/sam2-l_thr0.75_nms0.3_maxMask400.json'
rp_path = '/home/zkcs/zyj/label-engine/region_proposals/xBD/xBD_building/sam2-l_grid32_thr0.6_nms0.3_maxMask1000.json'
unseen_classes = None

# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/owlv2/4o_owl-b_ovd_xBD_building_coco.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/xBD_building_512/coco_results/lae_dino.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/xBD/coco_preds/gpt-4o-2024-11-20_ov_finetune_sam2-nms0.5-con0.75_georsclip_crop0.json'
coco_results_raw_path = None

# dataset = 'AerialMaritimeDomain'
# img_dir = '/home/zkcs/datasets/AerialMaritimeDrone/large_total/images'
# ann_path = '/home/zkcs/datasets/AerialMaritimeDrone/large_total/annotation.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/AerialMaritimeDrone/coco_preds/gpt-4o-2024-11-20_fintune_sam2-nms0.5-con0.75_georsclip_crop0.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/results/AerialMaritimeDrone/lae_dino.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/owlv2/4o_owl-b_ovd_AerialMaritimeDrone_coco.json'
# coco_results_raw_path = '/home/zkcs/zyj/label-engine/baseline/CastDet/AerialMaritimeDrone_predictions_oed.json'

threshold_search = False
if threshold_search:
    thresholds = [round(x, 2) for x in list(np.arange(0.0, 0.6, 0.02))]

save_results = True
unseen_classes = None

confidence = 0.75   # Confidence threshold for region proposals
min_crop_width = 0
clip_model = 'georsclip'
sam_model = 'sam2-grid32-max1000'

count = count_dir.split('/')[-1]

if save_results:
    save_preds_path = f'./results/{dataset}/ovd/coco_preds/{count}_{sam_model}-con{confidence}_{clip_model}.json'
    save_results_path_bbox = f'./results/{dataset}/ovd/{count}_{sam_model}-con{confidence}_{clip_model}_bbox.txt'
    save_results_path_segm = f'./results/{dataset}/ovd/{count}_{sam_model}-con{confidence}_{clip_model}_segm.txt'
    save_results_path_mIoU = f'./results/{dataset}/ovd/{count}_{sam_model}-con{confidence}_{clip_model}_mIoU.txt'
    print(f'Saving predictions to {save_preds_path}\n')
else:
    save_preds_path = None
    save_results_path_bbox = None
    save_results_path_segm = None
    save_results_path_mIoU = None

with open(ann_path, 'r') as f:
    val_anns = json.load(f)

if coco_results_raw_path is None:
    with open(rp_path, 'r') as f:
        rp_preds = json.load(f)
    print(device)
    model, tokenizer, preprocess = init_clip_model(clip_model, device=device)

    predictor = InstructSAM(val_anns, img_dir, count_dir, rp_preds)
    predictor.calculate_vocab_text_features(model, tokenizer)

    # Store predictions for COCO evaluation
    boxes_final_list = []
    labels_final_list = []
    segmentations_final_list = []
    img_name_list = [img['file_name'] for img in val_anns['images']]

    for image_name in tqdm(img_name_list):
        image_path = os.path.join(img_dir, image_name)
        predictor.set_image(image_path)
        predictor.load_rps_and_cnts(thr=confidence)
        predictor.calculate_pred_text_features(model, tokenizer, use_vocab=True)
        predictor.match_boxes_and_labels(model, preprocess, min_crop_width=min_crop_width, batch_size=200)
        if not predictor.boxes_final:
            print(f"No boxes_final for {image_name}")
        labels_final_list.append(predictor.labels_final)
        boxes_final_list.append(predictor.boxes_final)
        segmentations_final_list.append(predictor.segmentations_final)

    # Convert predictions to COCO format
    coco_predictions = convert_predictions_to_coco(
        img_name_list, boxes_final_list, labels_final_list, segmentations_final_list,
        scores_final_list=None, annotations=val_anns, save_path=save_preds_path)

else:
    with open(coco_results_raw_path, 'r') as f:
        coco_predictions = json.load(f)



###################  Calculate mF1 for each threshold (Optional, for traditional detectors)  ###################
if threshold_search:
    from instruct_sam.metrics import calculate_f1_scores
    import copy
    import numpy as np
    # Thresholds to evaluate
    # thresholds = [round(x, 4) for x in list(np.arange(0.15, 0.3, 0.0025))]
    mf1_values = []

    # Evaluate mAP50 for each threshold
    coco_predictions_copy = copy.deepcopy(coco_predictions)
    for threshold in tqdm(thresholds, desc="Evaluating thresholds"):
        coco_predictions_filtered = [pred for pred in coco_predictions_copy if pred['score'] >= threshold]
        # print(len(coco_predictions_filtered))
        # Calculate mF1
        class_metrics, mean_f1_score, _ = calculate_f1_scores(val_anns, coco_predictions_filtered)
        mf1_values.append(mean_f1_score)
        print(f'threshold: {threshold}, mF1: {mean_f1_score}')

    # calculate the best threshold
    best_threshold = thresholds[np.argmax(mf1_values)]
    print(f'best threshold: {best_threshold}, mF1: {mf1_values[np.argmax(mf1_values)]}')
    coco_predictions = [pred for pred in coco_predictions if pred['score'] >= best_threshold]
    # set the score to 1
    for pred in coco_predictions:
        pred['score'] = 1.0

###################  Evaluate mIoU, AP and F1  ###################

evaluate_ap_f1(val_anns, coco_predictions, iouType='bbox', unseen=unseen_classes, save_path=save_results_path_bbox)

# evaluate_ap_f1(val_anns, coco_predictions, iouType='segm', unseen=unseen_classes, save_path=save_results_path_segm)

# evaluate_mIoU(val_anns, coco_predictions, save_path=save_results_path_mIoU)

from instruct_sam.counting import eval_detection_counting_coco
cnt_metrics = eval_detection_counting_coco(coco_predictions, val_anns)
