import os
import json
from instruct_sam.matching import init_clip_model
from instruct_sam.instruct_sam import InstructSAM
from tqdm import tqdm
import time
import argparse
import yaml
import torch
import torch.multiprocessing as mp
from collections import defaultdict
import logging

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_images(rank, image_list, img_dir, args, result_queue, gpu_id):
    """在指定GPU上处理图像列表"""
    # 设置GPU（必须在子进程中设置）
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'
    
    print(f"Process {rank} using GPU {gpu_id}, processing {len(image_list)} images")
    
    # 重新加载所有必要的数据（不能从父进程继承CUDA上下文）
    with open(args.dataset_config, 'r') as f:
        dataset_config = json.load(f)
    dataset_info = dataset_config[args.dataset_name]
    ann_path = dataset_info['ann_path']
    with open(ann_path, 'r') as f:
        anns_coco = json.load(f)

    # 创建图像名到ID的映射
    img_name_to_id = {img['file_name']: img['id'] for img in anns_coco['images']}

    # 加载checkpoint配置
    with open(args.checkpoint_config, 'r') as f:
        checkpoint_config = yaml.load(f, Loader=yaml.FullLoader)
    ckpt_path = checkpoint_config[args.clip_model]
    
    print(f'Process {rank} loading {args.clip_model} checkpoint from {ckpt_path} to {device}.')
    model, tokenizer, preprocess = init_clip_model(
        args.clip_model, device, ckpt_path)

    with open(args.rp_path, 'r') as f:
        rp_preds = json.load(f)

    # 初始化预测器
    predictor = InstructSAM(anns_coco, img_dir, args.count_dir, rp_preds)

    # Use vocab feature cache for open vocabulary setting
    if args.setting == 'open_vocabulary':
        use_vocab = True
        predictor.calculate_vocab_text_features(model, tokenizer)
    else:
        use_vocab = False

    local_results = []
    
    for image_name in tqdm(image_list, desc=f'GPU {gpu_id}', position=rank, leave=False):
        try:
            image_path = os.path.join(img_dir, image_name)
            predictor.set_image(image_path)
            predictor.load_rps_and_cnts()
            
            start_time = time.time()
            predictor.calculate_pred_text_features(
                model, tokenizer, use_vocab=use_vocab)
            predictor.match_boxes_and_labels(
                model, preprocess, zero_count_warning=False)

            for label, bbox, segmentation, score in zip(predictor.labels_final, 
                                                     predictor.boxes_final, 
                                                     predictor.segmentations_final, 
                                                     predictor.scores_final):
                img_id = img_name_to_id.get(image_name)
                
                if img_id is not None:
                    prediction = {
                        "image_id": img_id,
                        "bbox": bbox,
                        "score": score,
                        "segmentation": segmentation,
                        "label": label
                    }
                    if args.setting == 'open_vocabulary':
                        prediction['category_id'] = predictor.category_name_to_id[label]
                    local_results.append(prediction)
        except Exception as e:
            print(f"Error processing image {image_name} on GPU {gpu_id}: {str(e)}")
            continue

    # 将结果放入队列
    result_queue.put((rank, local_results))
    print(f"GPU {gpu_id} finished processing {len(local_results)} predictions")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='dior_val')
    parser.add_argument('--dataset_config', type=str,
                        default='datasets/config.json')
    parser.add_argument('--checkpoint_config', type=str,
                        default='checkpoints/config.yaml')
    parser.add_argument('--count_dir', type=str,
                        default='./object_counts/dior_val/Qwen_open_ended')
    parser.add_argument('--rp_path', type=str,
                        default='./region_proposals/dior_val/sam2_hiera_l.json')
    parser.add_argument('--setting', type=str, default='open_ended',
                        choices=['open_vocabulary', 'open_ended', 'open_subclass'])
    parser.add_argument('--clip_model', type=str, default='remoteclip-b32')
    parser.add_argument('--num_gpus', type=int, default=None, help='Number of GPUs to use')
    args = parser.parse_args()

    return args

def main():
    setup_logging()
    args = parse_args()

    # 设置multiprocessing启动方法
    mp.set_start_method('spawn', force=True)

    # 检测GPU数量
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        if args.num_gpus is None:
            args.num_gpus = total_gpus
        args.num_gpus = min(args.num_gpus, total_gpus)
        print(f"Detected {total_gpus} GPUs, using {args.num_gpus} GPUs")
    else:
        args.num_gpus = 1
        print("CUDA not available, using CPU")

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

    img_name_list = [img['file_name'] for img in anns_coco['images']]
    print(f'Total images: {len(img_name_list)}')

    # 分割图像列表给不同GPU
    images_per_gpu = len(img_name_list) // args.num_gpus
    remainder = len(img_name_list) % args.num_gpus
    
    gpu_image_lists = []
    start_idx = 0
    for i in range(args.num_gpus):
        end_idx = start_idx + images_per_gpu
        if i < remainder:
            end_idx += 1
        gpu_image_lists.append(img_name_list[start_idx:end_idx])
        start_idx = end_idx

    print("Image distribution:")
    for i, img_list in enumerate(gpu_image_lists):
        print(f"  GPU {i}: {len(img_list)} images")

    # 创建结果队列
    result_queue = mp.Queue()
    
    # 启动多进程
    processes = []
    for rank in range(args.num_gpus):
        if len(gpu_image_lists[rank]) > 0:  # 只为有图像的GPU启动进程
            p = mp.Process(target=process_images, 
                          args=(rank, gpu_image_lists[rank], img_dir, args, result_queue, rank))
            p.start()
            processes.append(p)

    # 收集所有进程的结果
    all_results = []
    for _ in range(len(processes)):
        rank, results = result_queue.get()
        all_results.extend(results)
        print(f"Received {len(results)} results from process {rank}")

    # 等待所有进程结束
    for p in processes:
        p.join()

    # 保存最终结果
    os.makedirs(os.path.dirname(save_preds_path), exist_ok=True)
    with open(save_preds_path, 'w') as f:
        json.dump(all_results, f)

    print(f'Saved {args.setting} predictions to {save_preds_path}')
    print(f'Total predictions: {len(all_results)}')

if __name__ == '__main__':
    main()