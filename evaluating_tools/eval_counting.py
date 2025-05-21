from instruct_sam.counting import eval_gpt_counting, eval_detection_counting_coco
import json
from instruct_sam.metrics import get_cat_map, get_all_counts, convert_coco_raw_to_coco_pred
from instruct_sam.matching import init_clip_model
from instruct_sam.instruct_sam import InstructSAM
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count_dir', type=str, default=None,
                        help='The directory to the counting results.')
    parser.add_argument('--coco_pred_path', type=str, default=None,
                        help='The path to the coco prediction file.')
    parser.add_argument('--dataset_name', type=str, default='dior_mini')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
    parser.add_argument('--setting', type=str, required=True,
                        choices=['open_vocabulary',
                                 'open_ended', 'open_subclass'],
                        help='The setting of instruction.')
    parser.add_argument('--checkpoint_config', type=str,
                        default='./checkpoints/config.yaml')
    parser.add_argument('--map_model', type=str, default='georsclip',
                        choices=['georsclip', 'georsclip-b32', 'dfn2b',
                                 'remoteclip', 'remoteclip-b32',
                                 'skyclip', 'skyclip-b32'])
    parser.add_argument('--extra_classes', type=str,
                        choices=['unseen_classes',
                                 'means_of_transport', 'sports_field'],
                        default=None, help='Specify the extra classes to calculate metrics for.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--score_threshold', type=float, default=0.0,
                        help='Score threshold for filtering predictions.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    # dataset info
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[args.dataset_name]
    ann_path = dataset_info['ann_path']
    with open(ann_path, 'r') as f:
        anns_coco = json.load(f)

    extra_classes = args.extra_classes
    if extra_classes is not None:
        extra_classes = dataset_info[extra_classes]

    # priority: counting results > coco predictions
    if args.count_dir is not None:  # evaluate by counting results
        counting_results = args.count_dir
        eval_by_counting = True
    elif args.coco_pred_path is not None:  # evaluate by coco predictions
        counting_results = args.coco_pred_path
        with open(counting_results, 'r') as f:
            coco_preds_raw = json.load(f)
            counting_results = coco_preds_raw
        eval_by_counting = False
    else:
        raise ValueError(
            'Please provide either a counting results directory or a coco predictions file.')


    if args.setting != 'open_vocabulary':
        # Load checkpoint config
        with open(args.checkpoint_config, 'r') as f:
            checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
        ckpt_path = checkpoint_config[args.map_model]
        print(
            f'Loading {args.map_model} checkpoint from {ckpt_path} to {args.device}.')
        model, tokenizer, preprocess = init_clip_model(
            args.map_model, args.device, ckpt_path)

        predictor = InstructSAM(ann_path)
        all_counts = get_all_counts(counting_results)
        pred2cat = get_cat_map(
            model, tokenizer, predictor.categories, all_counts, threshold=0.95)

        print(f"Total unique classes: {len(all_counts)}")
        print(f"Examples of counts: {list(all_counts.items())[:5]}")

        if not eval_by_counting:
            coco_predictions = convert_coco_raw_to_coco_pred(
                coco_preds_raw, pred2cat, anns_coco, det_threshold=args.score_threshold)
    else:
        coco_predictions = coco_preds_raw if not eval_by_counting else None
        pred2cat = None

    if eval_by_counting:
        print('\n## Evaluating counting F1-score from {} ##'.format(args.count_dir))
        gpt_metrics = eval_gpt_counting(
            args.count_dir, ann_path, pred2cat, unseen=extra_classes)
    else:
        print('\n## Evaluating counting F1-score from {} ##'.format(args.coco_pred_path))
        cnt_metrics = eval_detection_counting_coco(
            coco_predictions, anns_coco, unseen=extra_classes, print_result=True)
