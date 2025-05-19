from instruct_sam.metrics import calculate_rp_recall
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_proposals', type=str, required=True,
                        help='The path to the mask proposals in coco format.')
    parser.add_argument('--dataset_name', type=str, default='dior_mini')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
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

    # load region proposals
    with open(args.mask_proposals, 'r') as f:
        mask_proposals = json.load(f)

    # calculate recall of class-agnostic masks
    calculate_rp_recall(anns_coco, mask_proposals,
                        score_threshold=args.score_threshold)
