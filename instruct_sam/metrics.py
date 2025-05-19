import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torch
from tabulate import tabulate
from pycocotools import mask as cocomask
import json
import copy
from collections import defaultdict
import os


def convert_predictions_to_coco(img_name_list, boxes_final_list, labels_final_list, segmentations_final_list,
                                scores_final_list=None, annotations=None, save_path=None):
    """
    Convert predictions to COCO format for evaluation.
    
    :param img_name_list: List of image names (keys in annotations).
    :param boxes_final_list: List of predicted boxes for each image.
    :param labels_final_list: List of predicted labels for each image.
    :param annotations: coco format annotations.
    :return: List of predictions in COCO format.
    """
    category_name_to_id = {item['name']: item['id'] for item in annotations['categories']}
    # map image name to image id (name2id)
    name2id = {item['file_name']: item['id'] for item in annotations['images']}
    coco_predictions = []
    for idx, img_name in enumerate(img_name_list):
        image_id = name2id[img_name]  # Assuming file names are like "123.jpg"
        boxes = boxes_final_list[idx]
        labels = labels_final_list[idx]
        segmentations = segmentations_final_list[idx]
        scores = scores_final_list[idx] if scores_final_list is not None else [1.0] * len(boxes)
        
        for i, box in enumerate(boxes):
            x, y, w, h = box  # COCO format: [x_min, y_min, width, height]
            label = labels[i]
            if label in category_name_to_id:
                category_id = category_name_to_id[label]
            else:
                raise ValueError(f"Label '{label}' not found in categories.")
            coco_predictions.append({
                "image_id": int(image_id),
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "segmentation": segmentations[i],
                "score": scores[i]
            })
    if save_path is not None:
        # create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(coco_predictions, f)
    return coco_predictions


def filter_rp_by_threshold(rp_preds, threshold=0, mAPnc=True, coco_input=False):
    coco_predictions = []
    # 如果coco_input为True，则输入的是COCO格式的预测结果，否则输入的是预测结果字典
    # 如果coco_input为True, 则过滤掉小于阈值的筛选框; 并将其他阈值设置为1    
    if coco_input:
        for pred in rp_preds:
            pred_copy = copy.deepcopy(pred)
            if pred_copy['score'] >= threshold:
                if mAPnc:
                    pred_copy['score'] = 1
                coco_predictions.append(pred_copy)
    else:
        for img_name, preds in rp_preds.items():
            image_id = int(img_name.split('.')[0])  # Assuming file names are like "001.jpg"
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']

            for bbox, score, label in zip(bboxes, scores, labels):
                if score >= threshold:
                    if mAPnc:
                        score = 1
                    x1, y1, width, height = bbox
                    coco_predictions.append({
                        "image_id": image_id,
                        "category_id": label + 1,  # Assuming category IDs in COCO format start from 1
                        "bbox": [x1, y1, width, height],
                        "score": score  # Substitute with constant score for mAPnc
                    })
    return coco_predictions


def get_all_counts(results):
    """
    计算所有类别的数量
    """
    all_counts = defaultdict(int)
    if isinstance(results, str):        # chatgpt counding dirctory
        for filename in os.listdir(results):
            if filename.endswith('.json'):
                file_path = os.path.join(results, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for category, count in data.items():
                    # 确保count是整数
                    try:
                        count_value = int(count) if isinstance(count, str) else count
                        all_counts[category] += count_value
                    except (ValueError, TypeError) as e:
                        print(f"警告：文件 {filename} 中类别 {category} 的计数 '{count}' 不能被转换为整数。错误: {e}")
    elif isinstance(results, list):     # list, coco_raw format
        for item in results:
            label = item['label']
            if label in all_counts:
                all_counts[label] += 1
            else:
                all_counts[label] = 1
    else:
        raise ValueError(f"Unsupported results format: {type(results)}")
    return all_counts


def get_cat_map(model, tokenizer, categories, all_counts, threshold=0.95):
    """
    使用已初始化好的CLIP模型，根据给定阈值输出一个字典：
    key: 预测到的类别
    value: 相似度最高的NWPU类别(若相似度>=阈值)，若所有类别的相似度小于阈值则删除该预测类别。
    
    Example:
        all_counts = {}
        for img_name in img_name_list:
            predictor.set_image(os.path.join(img_dir, img_name))
            predictor.load_rps_and_cnts()
            for cat, count in predictor.pred_counts.items():
                all_counts[cat] = all_counts.get(cat, 0) + count
        pred2cat = get_cat_map(model, tokenizer, predictor.categories,
                                            all_counts, threshold=0.95)
    """
    device = next(model.parameters()).device
    # 准备NWPU类别文本
    nwpu_texts = [f"an image of a {cat}" for cat in categories]
    nwpu_tokens = tokenizer(nwpu_texts).to(device)

    # 计算NWPU类别文本embedding
    with torch.no_grad():
        nwpu_text_embeds = model.encode_text(nwpu_tokens)
        nwpu_text_embeds = nwpu_text_embeds / nwpu_text_embeds.norm(dim=-1, keepdim=True)

    # 预测类别
    predicted_cats = list(all_counts.keys())
    predicted_texts = [f"an image of a {pcat}" for pcat in predicted_cats]
    predicted_tokens = tokenizer(predicted_texts).to(device)

    # 计算预测类别文本embedding
    with torch.no_grad():
        predicted_text_embeds = model.encode_text(predicted_tokens)
        predicted_text_embeds = predicted_text_embeds / predicted_text_embeds.norm(dim=-1, keepdim=True)

    # 计算相似度矩阵 [len(predicted_cats), len(categories)]
    # 行对应预测类别，列对应NWPU类别
    similarities = predicted_text_embeds @ nwpu_text_embeds.T

    # 输出字典: 
    # key为预测类别，value为高于threshold时最高的NWPU类别，否则该预测类别不加入结果
    result = {}
    for i, pred_cat in enumerate(predicted_cats):
        row = similarities[i]
        max_val, max_idx = torch.max(row, dim=0)
        if max_val.item() >= threshold:
            result[pred_cat] = categories[max_idx.item()]

    return result

import openai


API_KEY = 'sk-RWrbcDyifkghniRiE1FeC79a9c1244F19063AaC7266c3dA2'
BASE_URL = "https://api.v3.cm/v1/"      # HK线路
BASE_URL = "https://api.aaai.vip/v1/"      # HK线路

# API_KEY = 'sk-1a1f32ba0d3349f5be5d1e93fb3e1db2'
# BASE_URL = "https://api.deepseek.com"

openai.api_key = API_KEY
openai.base_url = BASE_URL
from instruct_sam.counting import extract_dict_from_string

def create_completion(prompt, model_name, json_output=False):
    try:
        message = {
            "model": model_name,
            "max_tokens": 6000,
            "temperature": 0,
            "top_p": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
        }
        
        if json_output:
            message["response_format"] = {"type": "json_object"}
        
        print('Sending message...')
        response = openai.chat.completions.create(**message)
        response_content = response.choices[0].message.content
        # print(response_content)
        pred2cat = extract_dict_from_string(response_content)
        if pred2cat == -1:
            return response_content
        return pred2cat
            
    except Exception as e:
        print(f"请求发生异常: {e}")
        return {}

def convert_coco_raw_to_coco_pred(coco_results_raw, pred2cat, val_anns, det_threshold=0.0):
    """
    Args:
        coco_results_raw: list, coco_raw format
        pred2cat: dict, predicted category map
        val_anns: dict, coco format annotations
        det_threshold: float, detection threshold
    """
    # 转换 qwen_oed_raw 为 COCO 格式的预测结果
    coco_predictions = []
    annotation_id = 1

    # 从 val_anns 获取 categories 信息
    categories = val_anns['categories']
    # 创建类别名称到ID的映4
    category_name_to_id = {cat['name']: cat['id'] for cat in categories}

    for item in coco_results_raw:
        label = item['label']
        
        # 如果标签不在映射字典的key里，跳过这个框
        if label not in pred2cat:
            continue
        
        if item['score'] < det_threshold:
            continue
        
        # 获取映射后的类别
        mapped_category = pred2cat[label]
        
        # 如果映射的类别不在数据集的类别中，跳过
        if mapped_category not in category_name_to_id:
            continue
        
        # 获取类别ID
        category_id = category_name_to_id[mapped_category]
        
        # 构建 COCO 格式的预测
        coco_pred = {
            'image_id': item['image_id'],
            'category_id': category_id,
            'bbox': item['bbox'],  # [x, y, width, height]
            'id': annotation_id  # COCO 格式需要唯一的 ID
        }
        if 'score' in item:
            coco_pred['score'] = item['score']
        else:
            coco_pred['score'] = 1
        if 'segmentation' in item:
            coco_pred['segmentation'] = item['segmentation'] 
        
        coco_predictions.append(coco_pred)
        annotation_id += 1
    return coco_predictions


# def map_final_label_to_cats(pred2cat, labels_final, boxes_final, segmentations_final):
#     """
#     Args:
#         labels_final (list): a list of predicted label names (str)
#         boxes_final (list): a list of bounding boxes
#         segmentations_final (list): a list of segmentations
#         pred2cat (dict): a dictionary mapping predicted labels to category labels

#     Returns:
#         tuple: updated lists of labels, boxes, and segmentations
        
#     Use Example:
#         labels_final, boxes_final, segmentations_final = map_final_label_to_cats(pred2cat, labels_final, boxes_final, segmentations_final)
#     """
#     labels_final_copy = labels_final.copy()
    
#     for label in labels_final_copy:
#         if label not in pred2cat.keys():
#             index = labels_final.index(label)
#             del boxes_final[index]
#             del labels_final[index]
#             del segmentations_final[index]
#         else:
#             labels_final[labels_final.index(label)] = pred2cat[label]
    
#     return labels_final, boxes_final, segmentations_final




def calculate_iou(box1, box2, iouType='bbox', img_height=None, img_width=None):
    """
    Calculate IoU between two bounding boxes or segmentation masks.
    
    Args:
        box1, box2: [x, y, w, h] for 'bbox' or segmentation (RLE or polygon) for 'segm'
        iouType (str): 'bbox' or 'segm'
        img_height (int): Image height (required for segm if converting polygons)
        img_width (int): Image width (required for segm if converting polygons)
    
    Returns:
        float: IoU value
    """
    if iouType == 'bbox':
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    elif iouType == 'segm':
        try:
            # Ensure image dimensions are provided
            if img_height is None or img_width is None:
                raise ValueError("img_height and img_width must be provided for segm IoU")

            # Convert inputs to RLE if necessary
            if isinstance(box1, dict) and 'counts' in box1:  # RLE format
                mask1 = box1
            elif isinstance(box1, list):  # Polygon format
                # Flatten nested polygon list if necessary (e.g., [[x1,y1], [x2,y2]] or [[[x1,y1], ...]])
                if isinstance(box1[0], list) and len(box1[0]) > 0 and isinstance(box1[0][0], list):
                    box1 = box1[0]  # Take first polygon if multiple are provided
                mask1 = cocomask.frPyObjects(box1, img_height, img_width)[0]
            else:
                raise ValueError(f"Unsupported segmentation format for box1: {type(box1)}")

            if isinstance(box2, dict) and 'counts' in box2:  # RLE format
                mask2 = box2
            elif isinstance(box2, list):  # Polygon format
                if isinstance(box2[0], list) and len(box2[0]) > 0 and isinstance(box2[0][0], list):
                    box2 = box2[0]
                mask2 = cocomask.frPyObjects(box2, img_height, img_width)[0]
            else:
                raise ValueError(f"Unsupported segmentation format for box2: {type(box2)}")

            # Compute intersection and union
            inter_area = cocomask.area(cocomask.merge([mask1, mask2], intersect=True))
            union_area = cocomask.area(cocomask.merge([mask1, mask2], intersect=False))
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        except Exception as e:
            print(f"Error in segm IoU calculation: {str(e)}")
            print(f"box1: {box1}")
            print(f"box2: {box2}")
            return 0.0  # Return 0 on failure to avoid crashing
    
    else:
        raise ValueError("iouType must be 'bbox' or 'segm'")

def calculate_f1_scores(anns, coco_preds, iouType='bbox', unseen=None, iou_threshold=0.5):
    """
    Calculate F1 scores and additional metrics for each class using COCO format annotations and predictions.
    
    Args:
        anns (dict): COCO format ground truth annotations
        coco_preds (list): COCO format predictions
        iouType (str): Type of IoU calculation (default: 'bbox') or 'segm'
        unseen (list): List of unseen class names (default: None) e.g. ['ship', 'basketball_court', 'harbor']
        iou_threshold (float): IoU threshold for determining true positives (default: 0.5)
    
    Returns:
        tuple: (class_metrics dict, mean_f1_score float, mean_f1_unseen float)
            - class_metrics: {cat_name: {'tp': int, 'fp': int, 'fn': int, 'precision': float, 'recall': float, 'f1': float}}
            - mean_f1_score: Average F1 across all classes
            - mean_f1_unseen: Average F1 for unseen classes (or None if unseen is None)
    """
    gt_by_image = {}
    pred_by_image = {}
    categories = {cat['id']: cat['name'] for cat in anns['categories']}
    cat_ids = set(categories.keys())
    
    # Get image dimensions for segm IoU
    img_dims = {img['id']: (img['height'], img['width']) for img in anns['images']}

    for ann in anns['annotations']:
        img_id = ann['image_id']
        if img_id not in gt_by_image:
            gt_by_image[img_id] = {}
        cat_id = ann['category_id']
        if cat_id not in gt_by_image[img_id]:
            gt_by_image[img_id][cat_id] = []
        gt_by_image[img_id][cat_id].append(ann['bbox'] if iouType == 'bbox' else ann['segmentation'])

    for pred in coco_preds:
        img_id = pred['image_id']
        if img_id not in pred_by_image:
            pred_by_image[img_id] = {}
        cat_id = pred['category_id']
        if cat_id not in pred_by_image[img_id]:
            pred_by_image[img_id][cat_id] = []
        pred_by_image[img_id][cat_id].append(pred['bbox'] if iouType == 'bbox' else pred['segmentation'])

    tp = {cat_id: 0 for cat_id in cat_ids}
    fp = {cat_id: 0 for cat_id in cat_ids}
    fn = {cat_id: 0 for cat_id in cat_ids}

    for img_id in gt_by_image:
        gt_cats = gt_by_image.get(img_id, {})
        pred_cats = pred_by_image.get(img_id, {})
        height, width = img_dims.get(img_id, (None, None))

        for cat_id in cat_ids:
            gt_boxes = gt_cats.get(cat_id, [])
            pred_boxes = pred_cats.get(cat_id, [])

            matched_gt = set()
            for pred_box in pred_boxes:
                max_iou = 0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    iou = calculate_iou(pred_box, gt_box, iouType, height, width)
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx
                
                if max_iou >= iou_threshold:
                    tp[cat_id] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp[cat_id] += 1
            
            fn[cat_id] += len(gt_boxes) - len(matched_gt)

    for img_id in pred_by_image:
        if img_id not in gt_by_image:
            for cat_id in pred_by_image[img_id]:
                fp[cat_id] += len(pred_by_image[img_id][cat_id])

    # Calculate per-class metrics
    class_metrics = {}
    for cat_id in cat_ids:
        tp_val = tp[cat_id]
        fp_val = fp[cat_id]
        fn_val = fn[cat_id]
        precision = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        recall = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        class_metrics[categories[cat_id]] = {
            'tp': tp_val,
            'fp': fp_val,
            'fn': fn_val,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    # Calculate mean F1 score
    mean_f1_score = np.mean([metrics['f1'] for metrics in class_metrics.values()]) if class_metrics else 0.0

    # Calculate mean F1 for unseen classes
    mean_f1_unseen = None
    if unseen:
        unseen_f1_scores = [class_metrics.get(cat_name, {'f1': 0.0})['f1'] for cat_name in unseen]
        mean_f1_unseen = np.mean(unseen_f1_scores) if unseen_f1_scores else 0.0

    return class_metrics, mean_f1_score, mean_f1_unseen

def evaluate_ap_f1(ann, coco_predictions, iouType='segm', unseen=None, save_path=None, print_result=True):
    """
    Evaluate AP50 and F1 scores with detailed metrics using COCO annotations and predictions.
    
    Args:
        ann (dict): COCO format ground truth annotations
        coco_predictions (list): COCO format predictions
        iouType (str): Type of IoU calculation ('bbox' or 'segm', default: 'segm')
        unseen (list): List of unseen class names (default: ['ship', 'basketball_court', 'harbor'])
        save_path (str): Path to save results as txt file (default: None)
    
    Returns:
        None: Prints and optionally saves results
    """
    # Initialize COCO ground truth and predictions
    coco_gt = COCO()
    coco_gt.dataset = ann
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(coco_predictions)

    try:
        # Calculate AP using COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, iouType)
        coco_eval.params.iouThrs = np.array([0.5])
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    except Exception as e:
        print(f"Error in COCOeval: {str(e)}, try using bbox iouType")
        # Calculate AP using COCOeval
        coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval.params.iouThrs = np.array([0.5])
        
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
    # print("IoU thresholds before eval:", coco_eval.params.iouThrs)
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # print("IoU thresholds before summarize:", coco_eval.params.iouThrs)
    # coco_eval.summarize()

    # Extract per-class AP50
    class_ap = {}
    cat_ids = [cat['id'] for cat in ann['categories']]
    cat_names = {cat['id']: cat['name'] for cat in ann['categories']}
    if coco_eval.stats.size > 0:
        ap50_all = coco_eval.stats[0]  # mAP50 across all classes
        for idx, cat_id in enumerate(coco_eval.params.catIds):
            ap = coco_eval.eval['precision'][0, :, idx, 0, -1].mean()
            if not np.isnan(ap):
                class_ap[cat_names[cat_id]] = ap
            else:
                class_ap[cat_names[cat_id]] = 0.0
    else:
        ap50_all = 0.0
        for cat in ann['categories']:
            class_ap[cat['name']] = 0.0

    # Calculate mAP50 for unseen classes
    if unseen is not None:
        unseen_cat_ids = [cat['id'] for cat in ann['categories'] if cat['name'] in unseen]
        unseen_aps = [class_ap[cat_names[cat_id]] for cat_id in unseen_cat_ids if cat_id in cat_names]
        map50_unseen = np.mean(unseen_aps) if unseen_aps else 0.0

    # Calculate F1 scores and detailed metrics
    class_metrics, mean_f1_score, mean_f1_unseen = calculate_f1_scores(
        ann, coco_predictions, iouType=iouType, unseen=unseen, iou_threshold=0.5
    )

    # Prepare table in order of ann['categories']
    table_data = []
    for cat in ann['categories']:
        cat_name = cat['name']
        metrics = class_metrics.get(cat_name, {'tp': 0, 'fp': 0, 'fn': 0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
        ap_score = class_ap.get(cat_name, 0.0)
        table_data.append([
            cat_name,
            metrics['tp'],
            metrics['fp'],
            metrics['fn'],
            f"{metrics['precision'] * 100:.1f}",
            f"{metrics['recall'] * 100:.1f}",
            f"{metrics['f1'] * 100:.1f}",
            f"{ap_score * 100:.1f}"
        ])

    # Format output
    headers = ['Category', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'AP50']
    table_str = tabulate(table_data, headers=headers, tablefmt='grid')

    return_summary = {
        'class_metrics': class_metrics,
        'mF1': mean_f1_score*100,
        'mAP50': ap50_all*100,
    }
    
    if unseen is not None:
        summary_str = (
            f"mAP50: {ap50_all * 100:.1f}\n"
            f"mAP50 unseen: {map50_unseen * 100:.1f}\n"
            f"mF1: {mean_f1_score * 100:.1f}\n"
            f"mF1 unseen: {mean_f1_unseen * 100:.1f}"
        )
        return_summary['mAP50_unseen'] = map50_unseen*100
        return_summary['mF1_unseen'] = mean_f1_unseen*100
    else:
        summary_str = (
            f"mAP50: {ap50_all * 100:.1f}\n"
            f"mF1: {mean_f1_score * 100:.1f}"
        )

    full_output = f"{table_str}\n\n{summary_str}"

    # Print results
    if print_result:
        print(full_output)

    # Save to file if save_path is provided
    if save_path is not None:
        try:
            # create directory if not exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(full_output)
            print(f"Results saved to {save_path}")
        except Exception as e:
            print(f"Failed to save results: {str(e)}")
    
    return  return_summary


def evaluate_mIoU(ann, coco_predictions, save_path=None):
    """
    Evaluate per-class IoU and mIoU for instance segmentation results compared to semantic segmentation.
    
    Args:
        ann (dict): COCO format ground truth annotations with segmentation
        coco_predictions (list): COCO format predictions with segmentation
    
    Returns:
        None: Prints table of per-class IoU and mIoU
    """
    # Initialize COCO ground truth
    coco_gt = COCO()
    coco_gt.dataset = ann
    coco_gt.createIndex()

    # Load predictions
    coco_dt = coco_gt.loadRes(coco_predictions)

    # Get categories and image IDs
    categories = {cat['id']: cat['name'] for cat in ann['categories']}
    cat_ids = list(categories.keys())
    img_ids = coco_gt.getImgIds()

    # Initialize accumulators for intersection and union per class
    intersection = {cat_id: 0 for cat_id in cat_ids}
    union = {cat_id: 0 for cat_id in cat_ids}
    gt_area = {cat_id: 0 for cat_id in cat_ids}  # For classes with no predictions
    pred_area = {cat_id: 0 for cat_id in cat_ids}  # For classes with no ground truth

    # Process each image
    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        height, width = img_info['height'], img_info['width']

        # Load ground truth annotations for this image
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        anns = coco_gt.loadAnns(ann_ids)

        # Load predicted annotations for this image
        pred_ids = coco_dt.getAnnIds(imgIds=img_id)
        preds = coco_dt.loadAnns(pred_ids)

        # Aggregate masks by class
        gt_masks = {cat_id: [] for cat_id in cat_ids}
        pred_masks = {cat_id: [] for cat_id in cat_ids}

        # Ground truth masks
        for ann in anns:
            cat_id = ann['category_id']
            if 'segmentation' in ann:
                if isinstance(ann['segmentation'], dict):  # RLE
                    mask = ann['segmentation']
                else:  # Polygon
                    mask = cocomask.frPyObjects(ann['segmentation'], height, width)[0]
                gt_masks[cat_id].append(mask)

        # Predicted masks
        for pred in preds:
            cat_id = pred['category_id']
            if 'segmentation' in pred:
                if isinstance(pred['segmentation'], dict):  # RLE
                    mask = pred['segmentation']
                else:  # Polygon
                    mask = cocomask.frPyObjects(pred['segmentation'], height, width)[0]
                pred_masks[cat_id].append(mask)

        # Compute per-class IoU for this image
        for cat_id in cat_ids:
            gt_mask_list = gt_masks[cat_id]
            pred_mask_list = pred_masks[cat_id]

            # Merge all instance masks into a single class mask
            if gt_mask_list:
                gt_merged = cocomask.merge(gt_mask_list, intersect=False)  # Union of GT masks
            else:
                gt_merged = cocomask.encode(np.zeros((height, width), dtype=np.uint8, order='F'))

            if pred_mask_list:
                pred_merged = cocomask.merge(pred_mask_list, intersect=False)  # Union of pred masks
            else:
                pred_merged = cocomask.encode(np.zeros((height, width), dtype=np.uint8, order='F'))

            # Compute intersection and union
            inter = cocomask.area(cocomask.merge([gt_merged, pred_merged], intersect=True))
            gt_area_cat = cocomask.area(gt_merged)
            pred_area_cat = cocomask.area(pred_merged)
            union_cat = gt_area_cat + pred_area_cat - inter

            # Accumulate
            intersection[cat_id] += inter
            union[cat_id] += union_cat
            gt_area[cat_id] += gt_area_cat
            pred_area[cat_id] += pred_area_cat

    # Calculate IoU per class
    class_iou = {}
    for cat_id in cat_ids:
        if union[cat_id] > 0:
            iou = intersection[cat_id] / union[cat_id]
        else:
            # If no predictions and no ground truth, IoU is undefined; typically set to 0 or 1 depending on convention
            # Here, we'll use 0 unless there's ground truth with no predictions (then ignore in mIoU)
            iou = 0.0 if gt_area[cat_id] > 0 or pred_area[cat_id] > 0 else float('nan')
        class_iou[categories[cat_id]] = iou

    # Calculate mIoU (excluding classes with no GT or pred)
    valid_ious = [iou for iou in class_iou.values() if not np.isnan(iou)]
    miou = np.mean(valid_ious) if valid_ious else 0.0

    # Prepare table
    table_data = []
    for cat_name, iou in class_iou.items():
        iou_str = f"{iou:.3f}" if not np.isnan(iou) else "N/A"
        table_data.append([cat_name, iou_str])

    # Format and print output
    headers = ['Category', 'IoU']
    table_str = tabulate(table_data, headers=headers, tablefmt='grid')
    summary_str = f"mIoU: {miou:.3f}"
    full_output = f"{table_str}\n\n{summary_str}"

    print(full_output)
    if save_path is not None:
        # create directory if not exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(full_output)
        print(f"Results saved to {save_path}")


def calculate_num_rp(rp_dict, threshold=0.01):
    cnt = 0
    for key, value in rp_dict.items():
        # filter bboxes with score >= threshold
        for i in range(len(value['scores'])):
            if value['scores'][i] >= threshold:
                cnt += 1
    # print(f'{cnt} region proposals')
    return cnt


def calculate_rp_recall(gt_data: list, pred_data: dict, pred_data2: dict = None, score_threshold: float = 0.01):
    """
    计算一个或两个预测结果的recall值
    
    参数:
        gt_data (list): Ground truth annotations (COCO格式)
        pred_data (dict): 第一个预测结果 (格式为 {"img_name": {"bboxes": [[x_min, y_min, w, h], ...]}})
        pred_data2 (dict, optional): 第二个预测结果 
        score_threshold (float): 分数阈值
    """
    def calculate_single_recall(pred_data, gt_annotations):
        print(f'{calculate_num_rp(pred_data, score_threshold)} region proposals')
        category_stats = {cat['id']: {'tp': 0, 'total': 0} for cat in gt_data['categories']}
        
        # 初始化类别统计的total
        for ann in gt_data['annotations']:
            cat_id = ann['category_id']
            category_stats[cat_id]['total'] += 1

        # 处理每个GT图片的预测结果
        for img_info in gt_data['images']:
            img_name = img_info['file_name']
            img_id = str(img_info['id'])
            
            if img_name not in pred_data:
                continue

            # 检查gt_annotations中是否有该图片的标注
            if img_id not in gt_annotations:
                continue  # 跳过没有标注的图片

            # 重置gt_boxes的matched状态
            gt_boxes = copy.deepcopy(gt_annotations[img_id])
            predictions = pred_data[img_name]
            pred_boxes = predictions['bboxes']
            pred_scores = predictions.get('scores', [1.0] * len(pred_boxes))

            # 处理每个预测框
            for pred_box, pred_score in zip(pred_boxes, pred_scores):
                if pred_score < score_threshold:
                    continue
                    
                max_iou = 0
                best_match_idx = -1
                
                for i, gt_box in enumerate(gt_boxes):
                    if gt_box['matched']:
                        continue
                    iou = calculate_iou(pred_box, gt_box['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        best_match_idx = i

                if max_iou >= 0.5 and best_match_idx != -1:
                    gt_boxes[best_match_idx]['matched'] = True
                    cat_id = gt_boxes[best_match_idx]['category_id']
                    category_stats[cat_id]['tp'] += 1

        # 计算每个类别的recall
        category_recalls = {}
        for cat_id, stats in category_stats.items():
            cat_recall = stats['tp'] / stats['total'] if stats['total'] > 0 else 0
            cat_name = category_map[cat_id]
            category_recalls[cat_name] = cat_recall

        # 计算平均recall
        mean_recall = sum(category_recalls.values()) / len(category_recalls) if category_recalls else 0.0
        
        return category_recalls, mean_recall

    # 创建类别映射
    category_map = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # 创建gt_annotations
    gt_annotations = {}
    for ann in gt_data['annotations']:
        img_id = str(ann['image_id'])
        if img_id not in gt_annotations:
            gt_annotations[img_id] = []
        gt_annotations[img_id].append({
            'bbox': ann['bbox'],
            'category_id': ann['category_id'],
            'matched': False
        })

    # 计算第一个预测结果的recall
    category_recalls1, mean_recall1 = calculate_single_recall(pred_data, gt_annotations)

    # 如果有第二个预测结果，计算它的recall
    if pred_data2 is not None:
        category_recalls2, mean_recall2 = calculate_single_recall(pred_data2, gt_annotations)

    # 准备表格数据
    table_data = []
    for cat_name in category_map.values():
        row = [
            cat_name,
            f"{category_recalls1[cat_name] * 100:.1f}"
        ]
        if pred_data2 is not None:
            row.append(f"{category_recalls2[cat_name] * 100:.1f}")
        table_data.append(row)

    # 定义表头
    headers = ['Category', 'Recall 1']
    if pred_data2 is not None:
        headers.append('Recall 2')

    # 打印结果
    print("\nPer-category Results:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print(f"\nMean Recall 1: {mean_recall1 * 100:.1f}")
    if pred_data2 is not None:
        print(f"Mean Recall 2: {mean_recall2 * 100:.1f}")

    # return {
    #     'recall1': mean_recall1,
    #     'category_recalls1': category_recalls1,
    #     'recall2': mean_recall2 if pred_data2 is not None else None,
    #     'category_recalls2': category_recalls2 if pred_data2 is not None else None
    # }