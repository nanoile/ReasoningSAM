# InstructSAM

Yijie Zheng, Weijie Wu, Qingyun Li, Xuehui Wang, Xu Zhou, Aiai Ren, Jun Shen, Long Zhao, Guoqing Li, Xue Yang

[Paper] [Colab] [Dataset]

## Note
Delete author name, api, work path when submitting anonymous code.

## Installation

## Getting Started
First download the checkpoints of SAM2, Qwen2.5-VL, GeoRSCLIP.

See the example notebooks for more details.


## Inference
First, prepare your prompts in the folder
It is recommended to perform batch inference.
It is recommended to prepare the unannotated data in COCO format, just leave the 'annotations' field blank.

### Object Counting
The counting result for each image is saved in JSON format, follwing the structure:
```
{
    'category_name1': number1 (int),
    'category_name2': number2 (int),
}
```
- Counting using asinchronous API request.
1. Open-Vocabulary
```bash
python inference_tools/async_count.py --dataset_name dior_mini \
                                      --dataset_config datasets/config.json \
                                      --base_url https://api.gpt.ge/v1/ \
                                      --api_key sk-RWrbcDyifkghniRiE1FeC79a9c1244F19063AaC7266c3dA2 \
                                      --model gpt-4o-2024-11-20 \
                                      --prompt_path prompts/dior/open_vocabulary.txt
```
2. Open-Subclass
```bash
python inference_tools/async_count.py --dataset_name dior_mini \
                                      --dataset_config datasets/config.json \
                                      --base_url https://api.gpt.ge/v1/ \
                                      --api_key sk-t4HuM9zbrn67xPn5B49c2c6353714069883944D04f4885F0 \
                                      --model gpt-4o-2024-11-20 \
                                      --prompt_path prompts/dior/open_subclass_means_of_transports.txt
```

- Alternatively, use LVLM deployed locally to inference. e.g., Use `Qwen2.5-VL-7B-Instruct` to count objects.
```bash
python inference_tools/qwen_count.py --dataset_name dior_mini \
                                     --pretrained_model_name_or_path /home/zkcs/checkpoints/Qwen2.5-VL-7B-Instruct \
                                     --prompt_path prompts/dior/open_ended.txt
```

### Generate mask proposals using SAM2.

The format of region (mask) proposals. follows a simple
```bash
python inference_tools/propose_regions.py --dataset_name dior_mini \
                                          --dataset_config datasets/config.json \
                                          --sam2_checkpoint ../../checkpoints/sam2/sam2_hiera_large.pt \
                                          --sam2_cfg //home/zkcs/zyj/sam2/sam2/configs/sam2/sam2_hiera_l.yaml \
```

### Mask-Label matching.
explain coco / open_ended_coco format

```bash
# Open-Vocabulary
python inference_tools/mask_label_matching.py --dataset_name dior_mini \
                                              --dataset_config datasets/config.json \
                                              --checkpoint_config checkpoints/config.yaml \
                                              --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_vocabulary \
                                              --rp_path ./region_proposals/dior_mini/sam2_hiera_l.json \
                                              --clip_model georsclip \
                                              --setting open_vocabulary

# Open-Ended
python inference_tools/mask_label_matching.py --dataset_name dior_mini \
                                              --dataset_config datasets/config.json \
                                              --checkpoint_config checkpoints/config.yaml \
                                              --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_ended \
                                              --rp_path ./region_proposals/dior_mini/sam2_hiera_l.json \
                                              --clip_model georsclip \
                                              --setting open_ended

# Open-Subclass
python inference_tools/mask_label_matching.py --dataset_name dior_mini \
                                              --dataset_config datasets/config.json \
                                              --checkpoint_config checkpoints/config.yaml \
                                              --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_subclass_means_of_transports \
                                              --rp_path ./region_proposals/dior_mini/sam2_hiera_l.json \
                                              --clip_model georsclip \
                                              --setting open_subclass
```

## Evaluation
The IoU threshold to determine whether a predicted box/mask is TP is set at 0.5.

### Evaluating Object Counting
Object counting can be evaluated via counting results, or via detection/segmentation predictions.
- Use counting results
```bash
# Open-Vocabulary
python evaluating_tools/eval_counting.py --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_vocabulary \
                                         --dataset_name dior_mini \
                                         --setting open_vocabulary

# Open-Ended
python evaluating_tools/eval_counting.py --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_ended \
                                         --dataset_name dior_mini \
                                         --setting open_ended

# Open-Subclass
python evaluating_tools/eval_counting.py --count_dir object_counts/dior_mini/gpt-4o-2024-11-20_open_subclass_means_of_transports \
                                         --dataset_name dior_mini \
                                         --setting open_subclass \
                                         --extra_classes means_of_transport
```
- Use recognition results
```bash
# Open-Vocabulary
python evaluating_tools/eval_counting.py --coco_pred_path results/dior_mini/open_vocabulary/coco_preds/gpt-4o-2024-11-20_open_vocabulary_sam2_hiera_l_georsclip_preds_coco.json \
                                         --dataset_name dior_mini \
                                         --setting open_vocabulary

# Open-Ended
python evaluating_tools/eval_counting.py --coco_pred_path results/dior_mini/open_ended/coco_preds/gpt-4o-2024-11-20_open_ended_sam2_hiera_l_georsclip_preds_coco.json \
                                         --dataset_name dior_mini \
                                         --setting open_ended

# Open-Subclass
python evaluating_tools/eval_counting.py --coco_pred_path results/dior_mini/open_subclass/coco_preds/gpt-4o-2024-11-20_open_subclass_means_of_transports_sam2_hiera_l_georsclip_preds_coco.json \
                                         --dataset_name dior_mini \
                                         --setting open_subclass \
                                         --extra_classes means_of_transport
```

### Evaluating Recall of Mask Proposals
```
python evaluating_tools/eval_proposal_recall.py --mask_proposal region_proposals/dior_mini/sam2_hiera_l.json
                                                --dataset_name dior_mini
```

### Evaluating Object Detection and Segmentation
1. Open-Vocabulary setting.
```
python evaluating_tools/eval_recognition.py --predictions ./results/dior_mini/open_vocabulary/coco_preds/gpt-4o-2024-11-20_open_vocabulary_sam2_hiera_l_georsclip_coco_results.json \
                                            --dataset_name dior_mini \
                                            --setting open_vocabulary \
                                            --extra_class unseen_classes
```

2. Open-Ended setting.
```
python evaluating_tools/eval_recognition.py --predictions /home/zkcs/zyj/label-engine/results/dior/osd/coco_preds/gpt-4o-2024-11-20_transports_visible_sam2-con0.75_georsclip_raw.json \
                                            --dataset_name dior_val \
                                            --setting open_ended
```

3. Open-Subclass setting.
```
python evaluating_tools/eval_recognition.py --predictions /home/zkcs/zyj/label-engine/results/dior/osd/coco_preds/gpt-4o-2024-11-20_transports_visible_sam2-con0.75_georsclip_raw.json \
                                            --dataset_name dior_val \
                                            --setting open_subclass \
                                            --extra_class means_of_transport
```

### Evaluating Predictions with Confidential Score
To evaluate predictions with confidential scores, when the the confidence threshold is swept from 0 to 1 (step 0.02), the threshold maximizing mF1 across categories is selected, and the corresponding cusp score is reported. e.g.,
```
python evaluating_tools/eval_recognition.py --predictions results/dior_mini/open_vocabulary/coco_preds/gpt-4o-2024-11-20_open_vocabulary_sam2_hiera_l_georsclip_preds_coco.json \
                                            --dataset_name dior_mini \
                                            --setting open_subclass \
                                            --extra_class means_of_transport \
                                            --score_sweeping
```

