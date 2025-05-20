import os
import base64
import openai
import json
from tabulate import tabulate
from collections import defaultdict
import time
from PIL import Image
import os
from qwen_vl_utils import process_vision_info
import torch


def ask_qwen(model, processor, img_path, prompt, device='cuda:0', max_new_tokens=5000):
    # start timing
    start_time = time.time()
    trucated_flag = False
    qwen_output = {}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    try:
        # Inference: Generation of the output
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.01, do_sample=True)   # repetition_penalty=1.2
    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA Out of Memory Error: {e}")

        # Get image dimensions from the input image
        with Image.open(img_path) as img:
            width, height = img.size
            print(f"Input image dimensions: {width}x{height}")

        # Calculate and print visual token count
        text_token_count = processor(text=[text], padding=True, return_tensors="pt")[
            "input_ids"].shape[1]
        total_token_count = inputs["input_ids"].shape[1]
        image_token_count = total_token_count - text_token_count
        print(f"Text tokens: {text_token_count}")
        print(f"Total tokens: {total_token_count}")
        print(f"Visual tokens: {image_token_count}")
        raise

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    output_token_count = generated_ids_trimmed[0].shape[0]
    if output_token_count == max_new_tokens:
        trucated_flag = True
        print(
            f"The output may be truncated as the maximum token limit of {max_new_tokens} has been")

    end_time = time.time()

    qwen_output['process_time'] = end_time - start_time
    qwen_output['output_text'] = output_text[0]
    qwen_output['output_token_count'] = output_token_count
    qwen_output['trucated_flag'] = trucated_flag

    return qwen_output


def encode_image_as_base64(image_path):
    with open(image_path, "rb") as img_file:
        # Read the image and encode it
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
        # Add the appropriate MIME prefix (assuming the image is a PNG)
        return f"data:image/png;base64,{base64_string}"


def predict(text_prompt, encoded_image, top_p=1, temperature=1, model="gpt-4o-2024-11-20", json_output=False):
    if json_output:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image
                            }
                        }
                    ],
                }
            ],
            top_p=top_p,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
    else:
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": text_prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image
                            }
                        }
                    ],
                }
            ],
            top_p=top_p,
            temperature=temperature
        )
    return completion


def count_gt(annotations_data, img_name):
    """
    Count the number of targets in each category in the specified image.

    :param annotations_data: The annotation data in COCO format (contains images and annotations fields).
    :param img_name: The image file name.
    :return: A dictionary, where the key is the category name and the value is the number of targets.
    """

    category_dict = {cat['id']: cat['name']
                     for cat in annotations_data['categories']}
    # ini category_counts
    category_counts = {category: 0 for category in category_dict.values()}

    # Find the image_id corresponding to the image
    image_id = None

    for img_info in annotations_data['images']:
        if img_info['file_name'] == img_name:
            image_id = img_info['id']
            break

    if image_id is None:
        print(f"The annotation of {img_name} is not found")
        return category_counts

    # Iterate through the annotation data to count the targets in the image
    for ann in annotations_data['annotations']:
        if ann['image_id'] == image_id:
            category_name = category_dict[ann['category_id']]
            category_counts[category_name] += 1

    return category_counts


def extract_dict_from_string(response_string):
    """
    Extract JSON-formatted content from a ChatGPT API response string and parse it into a dictionary.

    Args:
        response_string (str): String containing JSON data (typically from API response).

    Returns:
        dict: Parsed dictionary containing the counts, or -1 if parsing fails.
    """
    try:
        # Find positions of first '{' and last '}'
        start_idx = response_string.find('{')
        end_idx = response_string.rfind('}')

        # Extract the JSON content between braces
        json_content = response_string[start_idx:end_idx + 1]

        # Parse the JSON content into a dictionary
        counts_pred = json.loads(json_content)
        return counts_pred
    except Exception as e:
        print(
            f"Error when extracting counts from string: {e} \n{response_string}")
        return -1


def print_counts(pred_counts, gt_counts):
    """
    Print the categories and their counts in the prediction and ground truth that are not zero.

    :param pred_counts: A dictionary, the prediction counting result.
    :param gt_counts: A dictionary, the ground truth counting result.
    """
    print("The categories and their counts in the prediction and ground truth that are not zero are as follows:\n")
    print(f"{'Category':<30} {'Pred Count (Pred)':<20} {'GT Count (GT)':<20}")
    print("-" * 70)

    # Get all possible categories
    all_categories = set(pred_counts.keys()).union(gt_counts.keys())

    for category in all_categories:
        pred_count = pred_counts.get(category, 0)
        gt_count = gt_counts.get(category, 0)

        # Only print the categories that are not zero
        if pred_count != 0 or gt_count != 0:
            print(f"{category:<30} {pred_count:<20} {gt_count:<20}")


def save_count(pred_counts, save_dir, img_name, print_output_path=True):
    """
    Save the predicted counts `pred_counts` to a JSON file.

    Args:
        pred_counts (dict): Dictionary containing the predicted counting results.
        save_dir (str): Directory path where the JSON file will be saved.
        img_name (str): Name of the image, used to generate the JSON filename.
        print_output_path (bool, optional): Whether to print the output path. Defaults to True.
    """
    try:
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Construct the output file path
        # Replace image extension with .json
        file_name = os.path.splitext(img_name)[0] + ".json"
        save_path = os.path.join(save_dir, file_name)

        # Write pred_counts to JSON file
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump(pred_counts, json_file, indent=4, ensure_ascii=False)

        if print_output_path:
            print(f"Predicted counts saved to: {save_path}")
    except Exception as e:
        print(f"Error while saving predictions for {img_name}: {e}")


def load_gt_counts(gt_data):
    """
    Calculate the count of each category from the COCO format annotation file.

    :param gt_data: The path to the COCO format annotation file.
    :return: A dictionary, where the key is the category and the value is the count.
    """
    if isinstance(gt_data, str):
        with open(gt_data, 'r') as f:
            gt_data = json.load(f)

    # Get the mapping from category ID to name
    categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
    gt_counts = defaultdict(int)

    # Iterate through all annotations and accumulate counts by category ID
    for ann in gt_data['annotations']:
        category_id = ann['category_id']
        category_name = categories[category_id]
        # Each annotation corresponds to one instance
        gt_counts[category_name] += 1
    total_images = len(gt_data['images'])
    print(f"Total GT images: {total_images}")
    return dict(gt_counts)


def eval_gpt_counting(gpt_results_dir, gt_ann_path, pred2cat=None, unseen=None):
    """
    Evaluate object counting performance using GPT results and calculate precision, recall, F1 for each category.

    Args:
        gpt_results_dir (str): Directory containing GPT prediction JSON files
        gt_ann_path (str): Path to COCO format ground truth annotations
        pred2cat (dict, optional): Mapping from prediction category names to ground truth category names
        unseen (list, optional): List of unseen categories to calculate separate metrics for

    Returns:
        dict: Dictionary containing metrics for each category and macro averages
    """
    # Initialize metrics storage
    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)

    # Load ground truth data
    with open(gt_ann_path, 'r') as f:
        gt_data = json.load(f)
        gt_file_names = list(img['file_name'] for img in gt_data['images'])
        gt_file_name_to_id = {img['file_name']: img['id']
                              for img in gt_data['images']}
        categories = {cat['id']: cat['name'] for cat in gt_data['categories']}
        if pred2cat is None:
            pred2cat = {cat: cat for cat in categories.values()}

    # Debug print for categories
    print(f"Categories in ground truth: {categories}")
    print(f"pred2cat mapping: {pred2cat}")

    # Process each prediction file
    for file in os.listdir(gpt_results_dir):
        if not file.endswith('.json'):
            continue

        img_suffix = gt_file_names[0].split('.')[-1]
        img_name = file.replace('json', f'{img_suffix}')

        if img_name not in gt_file_names:
            continue

        # Load predictions
        with open(os.path.join(gpt_results_dir, file), 'r') as f:
            try:
                pred_data = json.load(f)
            except:
                print(f"Error loading JSON from {file}")
                continue

        # Get ground truth counts
        image_id = gt_file_name_to_id[img_name]
        gt_counts = defaultdict(int)
        for ann in gt_data['annotations']:
            if ann['image_id'] == image_id:
                cat_name = categories[ann['category_id']]
                gt_counts[cat_name] += 1

        # Get prediction counts
        pred_counts = defaultdict(int)
        for cat, count in pred_data.items():
            if cat in pred2cat:
                mapped_cat = pred2cat[cat]
            else:
                continue

            if isinstance(count, (int, float)):
                # FIXED: Use += instead of = to accumulate counts
                pred_counts[mapped_cat] += int(count)

        # Calculate TP, FP, FN
        all_cats = set(gt_counts.keys()).union(set(pred_counts.keys()))
        for cat in all_cats:
            gt = gt_counts.get(cat, 0)
            pred = pred_counts.get(cat, 0)

            tp_dict[cat] += min(pred, gt)
            fp_dict[cat] += max(pred - gt, 0)
            fn_dict[cat] += max(gt - pred, 0)

    # Load total GT counts
    gt_total = load_gt_counts(gt_ann_path)

    # Calculate metrics
    categories_all = sorted(set(gt_total.keys()).union(tp_dict.keys()))
    table = []
    precisions, recalls, f1s = [], [], []

    # For unseen categories
    unseen_precisions, unseen_recalls, unseen_f1s = [], [], []

    for cat in categories_all:
        tp = tp_dict.get(cat, 0)
        fp = fp_dict.get(cat, 0)
        fn = fn_dict.get(cat, 0)
        gt = gt_total.get(cat, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Add to unseen metrics if category is in unseen list
        if unseen and cat in unseen:
            unseen_precisions.append(precision)
            unseen_recalls.append(recall)
            unseen_f1s.append(f1)

        table.append([
            cat,
            gt,
            tp,
            fp,
            fn,
            f"{precision*100:.1f}",
            f"{recall*100:.1f}",
            f"{f1*100:.1f}"
        ])

    # Calculate macro averages
    mPrecision = sum(precisions) / len(precisions) if precisions else 0
    mRecall = sum(recalls) / len(recalls) if recalls else 0
    mF1 = sum(f1s) / len(f1s) if f1s else 0

    # Calculate unseen category averages
    unseen_metrics = {}
    if unseen and any(cat in categories_all for cat in unseen):
        u_precision = sum(unseen_precisions) / \
            len(unseen_precisions) if unseen_precisions else 0
        u_recall = sum(unseen_recalls) / \
            len(unseen_recalls) if unseen_recalls else 0
        u_f1 = sum(unseen_f1s) / len(unseen_f1s) if unseen_f1s else 0

        unseen_metrics = {
            'Precision': u_precision,
            'Recall': u_recall,
            'F1': u_f1
        }

    # Print results
    headers = ["Category", "GT Count", "TP",
               "FP", "FN", "Precision", "Recall", "F1"]
    count_model_names = gpt_results_dir.split('/')[-1]
    print(f"\nCounting Results for **{count_model_names}**")
    print(tabulate(table, headers=headers, tablefmt="grid"))
    print(f"\nMacro Average Precision: {mPrecision*100:.1f}%")
    print(f"Macro Average Recall: {mRecall*100:.1f}%")
    print(f"Macro Average F1: {mF1*100:.1f}%")

    # Print unseen category metrics if available
    if unseen_metrics:
        print(f"\nMean Metrics for Extra classes ({', '.join(unseen)})")
        print(f"Average Precision: {unseen_metrics['Precision']*100:.1f}%")
        print(f"Average Recall: {unseen_metrics['Recall']*100:.1f}%")
        print(f"Average F1: {unseen_metrics['F1']*100:.1f}%")

    result = {
        'per_category': {cat: dict(zip(
            ['GT Count', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1'],
            row[1:]
        )) for cat, row in zip(categories_all, table)},
        'macro_avg': {
            'Precision': mPrecision,
            'Recall': mRecall,
            'F1': mF1
        }
    }

    # Add unseen category metrics to result
    if unseen_metrics:
        result['unseen_avg'] = unseen_metrics

    return result


def eval_detection_counting_coco(pred_data, gt_data, unseen=None, print_result=True, thr=0.01):
    """
    Evaluate object counting performance using COCO format prediction results and calculate precision, 
    recall, F1 for each category.

    Args:
        pred_data (any): Path to COCO format prediction results JSON file or list of predictions
        gt_data (any): Path to COCO format ground truth annotations or annotation dict
        unseen (list): List of unseen categories to calculate separate metrics for
        print_result (bool): Whether to print the results table
        thr (float): Confidence threshold for detection filtering

    Returns:
        dict: Dictionary containing metrics for each category and macro averages (as percentages with 1 decimal place)
    """
    # Initialize metrics storage
    tp_dict = defaultdict(int)
    fp_dict = defaultdict(int)
    fn_dict = defaultdict(int)

    if isinstance(pred_data, str):
        with open(pred_data, 'r') as f:
            pred_data = json.load(f)

    if isinstance(gt_data, str):
        with open(gt_data, 'r') as f:
            gt_data = json.load(f)

    # Create category lookup
    category_dict = {cat['id']: cat['name'] for cat in gt_data['categories']}

    # Group predictions by image_id
    predictions_by_image = defaultdict(list)
    for pred in pred_data:
        if pred.get('score', 1.0) >= thr:  # Filter by threshold
            predictions_by_image[pred['image_id']].append(pred)

    # Group ground truth annotations by image_id
    gt_by_image = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_by_image[ann['image_id']].append(ann)

    # Process each image
    for image_id in set(gt_by_image.keys()).union(predictions_by_image.keys()):
        # Get ground truth counts
        gt_counts = defaultdict(int)
        for ann in gt_by_image.get(image_id, []):
            cat_name = category_dict[ann['category_id']]
            gt_counts[cat_name] += 1

        # Get prediction counts
        pred_counts = defaultdict(int)
        for pred in predictions_by_image.get(image_id, []):
            cat_name = category_dict[pred['category_id']]
            pred_counts[cat_name] += 1

        # Calculate TP, FP, FN for this image
        all_cats = set(gt_counts.keys()).union(set(pred_counts.keys()))
        for cat in all_cats:
            gt = gt_counts.get(cat, 0)
            pred = pred_counts.get(cat, 0)

            tp_dict[cat] += min(pred, gt)
            fp_dict[cat] += max(pred - gt, 0)
            fn_dict[cat] += max(gt - pred, 0)

    # Load total GT counts
    gt_total = load_gt_counts(gt_data)

    # Calculate metrics
    categories_all = sorted(set(gt_total.keys()).union(tp_dict.keys()))
    table = []
    precisions, recalls, f1s = [], [], []

    # For unseen categories
    unseen_precisions, unseen_recalls, unseen_f1s = [], [], []

    for cat in categories_all:
        tp = tp_dict.get(cat, 0)
        fp = fp_dict.get(cat, 0)
        fn = fn_dict.get(cat, 0)
        gt = gt_total.get(cat, 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision +
                                         recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # Add to unseen metrics if category is in unseen list
        if unseen and cat in unseen:
            unseen_precisions.append(precision)
            unseen_recalls.append(recall)
            unseen_f1s.append(f1)

        table.append([
            cat,
            gt,
            tp,
            fp,
            fn,
            f"{precision*100:.1f}",
            f"{recall*100:.1f}",
            f"{f1*100:.1f}"
        ])

    # Calculate macro averages
    mPrecision = sum(precisions) / len(precisions) if precisions else 0
    mRecall = sum(recalls) / len(recalls) if recalls else 0
    mF1 = sum(f1s) / len(f1s) if f1s else 0

    # Calculate unseen category averages
    unseen_metrics = {}
    if unseen and any(cat in categories_all for cat in unseen):
        u_precision = sum(unseen_precisions) / \
            len(unseen_precisions) if unseen_precisions else 0
        u_recall = sum(unseen_recalls) / \
            len(unseen_recalls) if unseen_recalls else 0
        u_f1 = sum(unseen_f1s) / len(unseen_f1s) if unseen_f1s else 0

        unseen_metrics = {
            'Precision': round(u_precision * 100, 1),
            'Recall': round(u_recall * 100, 1),
            'F1': round(u_f1 * 100, 1)
        }

    # Print results
    headers = ["Category", "GT Count", "TP",
               "FP", "FN", "Precision", "Recall", "F1"]
    if print_result:
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print(f"\nMacro Average Precision: {mPrecision*100:.1f}%")
        print(f"Macro Average Recall: {mRecall*100:.1f}%")
        print(f"Macro Average F1: {mF1*100:.1f}%")

        if unseen_metrics:
            print(f"\nMean Metrics for Extra classes ({', '.join(unseen)})")
            print(f"Average Precision: {unseen_metrics['Precision']}%")
            print(f"Average Recall: {unseen_metrics['Recall']}%")
            print(f"Average F1: {unseen_metrics['F1']}%")

    # Convert metrics to percentage with 1 decimal place
    result = {
        'per_category': {cat: {
            'GT Count': row[1],
            'TP': row[2],
            'FP': row[3],
            'FN': row[4],
            'Precision': round(float(row[5]), 1),
            'Recall': round(float(row[6]), 1),
            'F1': round(float(row[7]), 1)
        } for cat, row in zip(categories_all, table)},
        'macro_avg': {
            'Precision': round(mPrecision * 100, 1),
            'Recall': round(mRecall * 100, 1),
            'F1': round(mF1 * 100, 1)
        }
    }

    # Add unseen category metrics to result
    if unseen_metrics:
        result['unseen_avg'] = unseen_metrics

    return result
