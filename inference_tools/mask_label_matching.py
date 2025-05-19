import os
import json
from instruct_sam.matching import init_clip_model
from instruct_sam.instruct_sam import InstructSAM
from tqdm import tqdm
import time
import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='dior_mini')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
    parser.add_argument('--checkpoint_config', type=str,
                        default='checkpoints/config.yaml')
    parser.add_argument('--count_dir', type=str,
                        default='./object_counts/dior_mini/gpt-4o-2024-11-20_open_vocabulary')
    parser.add_argument('--rp_path', type=str,
                        default='./region_proposals/dior_mini/sam2_hiera_l.json')
    parser.add_argument('--setting', type=str, default='open_ended',
                        choices=['open_vocabulary', 'open_ended', 'open_subclass'])
    parser.add_argument('--clip_model', type=str, default='georsclip')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # dataset info
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[args.dataset_name]
    img_dir = dataset_info['img_dir']
    ann_path = dataset_info['ann_path']
    with open(ann_path, 'r') as f:
        anns_coco = json.load(f)

    # InstructSAM Components
    count = args.count_dir.split('/')[-1]
    sam_model = args.rp_path.split('/')[-1].split('.')[0]
    save_preds_path = f'./results/{args.dataset_name}/{args.setting}/coco_preds/{count}_{sam_model}_{args.clip_model}_preds_coco.json'
    print(f'Save {args.setting} predictions to {save_preds_path}')

    # Load checkpoint config
    with open(args.checkpoint_config, 'r') as f:
        checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
    ckpt_path = checkpoint_config[args.clip_model]
    print(
        f'Loading {args.clip_model} checkpoint from {ckpt_path} to {args.device}.')
    model, tokenizer, preprocess = init_clip_model(
        args.clip_model, args.device, ckpt_path)

    with open(args.rp_path, 'r') as f:
        rp_preds = json.load(f)
    img_name_list = [img['file_name'] for img in anns_coco['images']]

    predictor = InstructSAM(anns_coco, img_dir, args.count_dir, rp_preds)

    # Use vocab feature cache to accelerate text features calculation for open vocabulary setting
    if args.setting == 'open_vocabulary':
        use_vocab = True
        predictor.calculate_vocab_text_features(model, tokenizer)
    else:
        use_vocab = False

    coco_results_raw = []

    for image_name in tqdm(img_name_list):
        image_path = os.path.join(img_dir, image_name)
        predictor.set_image(image_path)
        predictor.load_rps_and_cnts()
        start_time = time.time()
        predictor.calculate_pred_text_features(
            model, tokenizer, use_vocab=use_vocab)
        predictor.match_boxes_and_labels(
            model, preprocess, zero_count_warning=False)

        for label, bbox, segmentation, score in zip(predictor.labels_final, predictor.boxes_final, predictor.segmentations_final, predictor.scores_final):
            prediction = {
                "image_id": predictor.img_id,
                "bbox": bbox,
                "score": score,
                "segmentation": segmentation,
                "label": label
            }
            if args.setting == 'open_vocabulary':
                prediction['category_id'] = predictor.category_name_to_id[label]
            coco_results_raw.append(prediction)

    os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)
    with open(save_preds_path, 'w') as f:
        json.dump(coco_results_raw, f)

    print(f'Saved {args.setting} predictions to {save_preds_path}')
