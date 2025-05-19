import os
import json
from instruct_sam.instruct_sam import InstructSAM
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='dior_mini')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
    parser.add_argument('--sam2_checkpoint', type=str, default='')
    parser.add_argument('--sam2_cfg', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_masks', type=int, default=200)
    parser.add_argument('--min_mask_region_area', type=int, default=0)
    parser.add_argument('--rp_save_path', type=str, default=None)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[args.dataset_name]
    img_dir = dataset_info['img_dir']
    ann_path = dataset_info['ann_path']

    if args.rp_save_path is None:
        sam_model = args.sam2_cfg.split('/')[-1].split('.')[0]
        rp_save_path = f'./region_proposals/{args.dataset_name}/{sam_model}.json'
    else:
        rp_save_path = args.rp_save_path

    with open(ann_path, 'r') as f:
        val_anns = json.load(f)

    img_name_list = [ann['file_name'] for ann in val_anns['images']]
    print(f'Total images: {len(img_name_list)}')

    sam2 = build_sam2(args.sam2_cfg, args.sam2_checkpoint,
                      device=args.device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(sam2, pred_iou_thresh=0.75,
                                                stability_score_thresh=0.75,
                                                points_per_side=24,
                                                crop_n_layers=1,
                                                box_nms_thresh=0.5,
                                                points_per_batch=256)

    predictor = InstructSAM(img_dir=img_dir)
    results_dict = {}

    for i, img_name in enumerate(tqdm(img_name_list)):
        image_path = os.path.join(img_dir, img_name)
        predictor.set_image(image_path)
        predictor.segment_anything(
            mask_generator, max_masks=args.max_masks, min_mask_region_area=args.min_mask_region_area)
        results_dict[img_name] = {
            "bboxes": predictor.bboxes,
            "labels": predictor.labels,
            "scores": predictor.scores,
            "segmentations": predictor.segmentations
        }

    # check rp_save_path, if its base dir not exist, create it
    if not os.path.exists(os.path.dirname(rp_save_path)):
        os.makedirs(os.path.dirname(rp_save_path))

    with open(rp_save_path, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Saved mask proposals to {rp_save_path}")
