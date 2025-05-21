from instruct_sam.counting import extract_dict_from_string, save_count, ask_qwen
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import json
import time
import logging
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instruction-Oriented Object Counting using Qwen2.5-VL.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Name of the dataset (e.g., 'dior_mini'). Must match a key in the config file.",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="datasets/config.json",
        help="Path to the dataset config file (default: datasets/config.json).",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default='prompts/dior/open_vocabulary.txt',
        required=True,
        help="Path to the prompt file.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Directory to the pretrained model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Device to run the model on (default: cuda:0).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to the save directory (default: ./object_counts).",
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # set device
    device = args.device

    dataset = args.dataset_name
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[dataset]
    ann_path = dataset_info['ann_path']
    img_dir = dataset_info['img_dir']

    with open(args.prompt_path, 'r') as f:
        prompt = f.read()

    save_dir = args.save_dir
    if save_dir is None:
        base_name = os.path.basename(args.prompt_path).split('.')[0]
        save_dir = f'./object_counts/{dataset}/Qwen_{base_name}'


    ################## DO NOT MODIFY BELOW THIS LINE ##################
    output_process_time_path = f'./object_counts/Qwen_{dataset}_{base_name}_runtime.json'
    # Create log file
    os.makedirs('./log', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f'./log/qwen_counting_{dataset}_{timestamp}.log'

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Prompt: {prompt}\nRunning on device: {device}")
    logger.info(f"Image directory: {img_dir}\nAnnotation path: {ann_path}")
    logger.info(f"save couning dir: {save_dir}")
    logger.info(f"output process time path: {output_process_time_path}")

    # load model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype="auto", device_map={"": device}
    )
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        args.pretrained_model_name_or_path, min_pixels=min_pixels, max_pixels=max_pixels)


    with open(ann_path, 'r') as f:
        annotations = json.load(f)

    # Define category mapping (assuming these are the categories from prompt_od)
    category_names = [cat["name"] for cat in annotations["categories"]]
    category_id_map = {name: idx + 1 for idx,
                    name in enumerate(category_names)}  # COCO IDs start at 1
    print(category_names, category_id_map)

    # Init list to record processing times
    process_time = {}

    img_name_list = [ann['file_name'] for ann in annotations['images']]

    for idx, image_name in tqdm(enumerate(img_name_list), total=len(img_name_list), desc="Processing images"):
        image_path = os.path.join(img_dir, image_name)
        img_array = np.array(Image.open(image_path))
        qwen_output = ask_qwen(model, processor, image_path,
                            prompt, device=device, max_new_tokens=5000)

        logger.info(f'Counting results of {image_name}')
        logger.info(f'{qwen_output["process_time"]}s')
        logger.info(qwen_output['output_text'])
        logger.info(f'output_token_count: {qwen_output["output_token_count"]}')

        pred_counts = extract_dict_from_string(qwen_output['output_text'])
        logger.info(f'pred_counts: {pred_counts}')

        if pred_counts == -1:
            bbox_cnt = 0
            print(
                f"Error in extracting counts from the output text for image {image_name}")
        else:
            bbox_cnt = sum(pred_counts.values())
            save_count(pred_counts, save_dir=save_dir,
                    img_name=image_name, print_output_path=False)
        process_time[image_name] = {
            'time': qwen_output['process_time'],
            'bbox_cnt': bbox_cnt,
            'token_cnt': qwen_output['output_token_count']
        }

    # dump process_time
    with open(output_process_time_path, 'w') as f:
        json.dump(process_time, f, indent=4)

    # print current time
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print("Done!")
