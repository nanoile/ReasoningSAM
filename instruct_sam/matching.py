import numpy as np
import torch
from PIL import Image
import os
import json
from instruct_sam.visualize import visualize_prediction
# from instruct_sam.visualize import CATEGORIES
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import transforms
import pulp
import open_clip
import time
from transformers import AutoProcessor, AutoModel


def get_preprocess_rs5m(image_resolution=224):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
    )
    preprocess_val = transforms.Compose([
        transforms.Resize(
            size=image_resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(image_resolution), 
        transforms.ToTensor(),
        normalize,
    ])
    return preprocess_val


def init_clip_model(clip_model, device='cuda:0', ckpt_path=None):
    """
    clip_model: 'dfn2b', 'georsclip', 'remoteclip', 'skyclip'
    return: model, tokenizer, preprocess
    """
    if ckpt_path is None:
        raise ValueError(f"ckpt_path is None for {clip_model}")
    if clip_model == 'dfn2b':
        # ViT-L/14 (DFN)
        model_name = 'ViT-L-14' # ('ViT-L-14', 'dfn2b_s39b'),
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_path)
        tokenizer = open_clip.get_tokenizer(model_name)
    
    elif clip_model == 'remoteclip':
        model_name = 'ViT-L-14'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        message = model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer(model_name)

    elif clip_model == 'remoteclip-b32':
        model_name = 'ViT-B-32'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=ckpt_path)
        tokenizer = open_clip.get_tokenizer(model_name)

    elif clip_model == 'georsclip':
        model, _, _ = open_clip.create_model_and_transforms("ViT-L/14", pretrained=ckpt_path)
        preprocess = get_preprocess_rs5m()
        tokenizer = open_clip.tokenize
    
    elif clip_model == 'georsclip-b32':
        # GeoRSCLIP
        model, _, _ = open_clip.create_model_and_transforms("ViT-B/32", pretrained=ckpt_path)
        preprocess = get_preprocess_rs5m()
        tokenizer = open_clip.tokenize
    
    elif clip_model == 'skyclip':
        # SkyCLIP
        from third_party.open_clip.factory import create_skyclip_model, get_skyclip_tokenizer
        model_arch_name = 'ViT-L-14'
        model, _, preprocess = create_skyclip_model(model_arch_name,
                                                            ckpt_path,
                                                            precision='amp',
                                                            # device=device,
                                                            output_dict=True,
                                                            force_quick_gelu=True)
        tokenizer = get_skyclip_tokenizer(model_arch_name)
        from third_party.open_clip.model import CLIP as ThirdPartyClip
        ThirdPartyClip.__bases__ = (open_clip.model.CLIP,) + ThirdPartyClip.__bases__
    
    elif clip_model == 'skyclip-b32':
        # SkyCLIP
        from third_party.open_clip.factory import create_skyclip_model, get_skyclip_tokenizer
        model_arch_name = 'ViT-B-32'
        model, _, preprocess = create_skyclip_model(model_arch_name,
                                                            ckpt_path,
                                                            precision='amp',
                                                            # device=device,
                                                            output_dict=True,
                                                            force_quick_gelu=False)
        tokenizer = get_skyclip_tokenizer(model_arch_name)
        from third_party.open_clip.model import CLIP as ThirdPartyClip
        ThirdPartyClip.__bases__ = (open_clip.model.CLIP,) + ThirdPartyClip.__bases__
  
    else:
        raise ValueError(f"Invalid clip model: {clip_model}")

    model.cuda(device).eval();
    return model, tokenizer, preprocess

def filter_boxes(boxes, segmentations, scores, labels, threshold):
    """
    根据分数阈值过滤边界框。如果过滤后留下的框数量小于 min_box，则保留置信度最高的 min_box 个框。
    :param boxes: 边界框坐标，形状为 [num_boxes, 4]
    :param scores: 边界框分数，形状为 [num_boxes]
    :param labels: 边界框标签，形状为 [num_boxes]
    :param threshold: 分数阈值
    :return: 过滤后的边界框坐标、分数和标签
    """
    # 将所有输入转换为 numpy 数组
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)
    
    if segmentations is None:
        segmentations = boxes
    else:
        # 保持 segmentations 作为列表，因为每个分割掩码可能有不同数量的点
        segmentations = list(segmentations)

    # 使用布尔索引过滤
    mask = scores >= threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    filtered_segmentations = [segmentations[i] for i in range(len(segmentations)) if mask[i]]

    return filtered_boxes.tolist(), filtered_scores.tolist(), filtered_labels.tolist(), filtered_segmentations


def assign_labels_with_constraints(similarity_scores, gpt_predicted_counts):
    """
    Assign region proposals to labels based on similarity scores and constraints.

    Args:
        similarity_scores: np.ndarray of shape (N, M), similarity scores between regions and labels.
        gpt_predicted_counts: dict, number of regions to assign for each label (e.g., {"harbor": 1, "ship": 4}).

    Returns:
        assignments: list of tuples (region_index, label_index), the optimal assignment.
    """
    # Step 1: Problem setup
    num_regions, num_labels = similarity_scores.shape
    cost_matrix = 1 - similarity_scores  # Convert similarity scores to cost

    # Initialize the ILP problem (minimization problem)
    prob = pulp.LpProblem("RegionProposalAssignment", pulp.LpMinimize)

    # Step 2: Define variables
    # X[i, j] = 1 if region i is assigned to label j, else 0
    X = pulp.LpVariable.dicts(
        "X", 
        ((i, j) for i in range(num_regions) for j in range(num_labels)), 
        cat="Binary"
    )

    # Step 3: Define the objective function
    # Minimize the total cost: sum(D[i][j] * X[i][j])
    prob += pulp.lpSum(cost_matrix[i, j] * X[i, j] for i in range(num_regions) for j in range(num_labels))

    # Step 4: Add constraints
    # Constraint 1: Each region proposal can be assigned to at most one label
    for i in range(num_regions):
        prob += pulp.lpSum(X[i, j] for j in range(num_labels)) <= 1

    # Constraint 2: Each label must be assigned to exactly the specified number of regions
    total_gpt_count = sum(gpt_predicted_counts.values())
    if num_regions <= total_gpt_count:
        # Case 1: Total regions are fewer than total GPT-predicted counts
        prob += pulp.lpSum(X[i, j] for i in range(num_regions) for j in range(num_labels)) == num_regions
        label_indices = list(gpt_predicted_counts.keys())
        for j, label in enumerate(label_indices):
            prob += pulp.lpSum(X[i, j] for i in range(num_regions)) <= gpt_predicted_counts[label]
    else:
        # Case 2: Use the original constraint for each label
        label_indices = list(gpt_predicted_counts.keys())
        for j, label in enumerate(label_indices):
            prob += pulp.lpSum(X[i, j] for i in range(num_regions)) == gpt_predicted_counts[label]

    # Step 5: Solve the ILP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    # prob.solve(pulp.PULP_CBC_CMD())

    # Step 6: Extract the results
    if pulp.LpStatus[prob.status] != "Optimal":
        print("Warning: No optimal solution found!")
        return []
        # raise ValueError("No optimal solution found!")

    assignments = [
        (i, j) 
        for i in range(num_regions) 
        for j in range(num_labels) 
        if pulp.value(X[i, j]) > 0.5
    ]
    
    return assignments


def match_boxes_and_counts(image, boxes, object_cnt, text_features,
                           model, preprocess, segmentations=None,
                           crop_scale=1.2, batch_size=200, full_categories=None, min_crop_width=0, 
                           show_similarities=False, zero_count_warning=True):
    """_summary_

    Args:
        image (np.array): img_array
        boxes (_type_): _description_
        object_cnt (_type_): _description_
        text_features (_type_): _description_
        model (_type_): _description_
        preprocess (_type_): _description_
        crop_scale (float, optional): _description_. Defaults to 1.2.
        min_crop_width (int, optional): _description_. Defaults to 0.
        show_similarities (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    device = next(model.parameters()).device

    # Step 1: 提取 GPT 预测的目标类别及其计数
    gpt_predicted_classes = [cls for cls, count in object_cnt.items() if count > 0]
    gpt_predicted_counts = {cls: count for cls, count in object_cnt.items() if count > 0}
    # print(gpt_predicted_counts)
    
    if not gpt_predicted_classes:
        if zero_count_warning:
            print("Warning: gpt_predicted_classes is empty. Returning empty results.")
        return [], [], [], []  # 返回空结果
    
    # Step 2: 裁剪 Region Proposal 区域并计算图像特征
    region_features = []
    regions = []
    # 获取图像的宽和高
    image_width, image_height = image.size

    for box in boxes:
        x, y, w, h = box

        if x > image_width or y > image_height:
            print(f'Warning: box {box} out of image size {image.size}')
            x = image_width - 2
            y = image_height - 2
            print("---------------")
        
        # 计算裁剪框的中心点
        center_x = x + w / 2
        center_y = y + h / 2

        # 计算裁剪框的长宽（根据 crop_scale 调整） the 5 min avg bbox areas range from 12 (whistle) to 94 (baterry) 
        new_w = max(min_crop_width, w * crop_scale)
        new_h = max(min_crop_width, h * crop_scale)

        # 计算裁剪框的左上角和右下角坐标
        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2


        # 确保裁剪框不会超出图像边界
        new_x1 = max(0, new_x1)  # 左边界
        new_y1 = max(0, new_y1)  # 上边界
        new_x2 = min(image_width, new_x2)  # 右边界
        new_y2 = min(image_height, new_y2)  # 下边界
            
        region = image.crop((new_x1, new_y1, new_x2, new_y2))
        if isinstance(model, open_clip.model.CLIP):
            regions.append(preprocess(region).unsqueeze(0))
        else:
            regions.append(preprocess(region))
    
    # 将所有 region 拼接成一个 batch
    regions_batch = torch.cat(regions, dim=0).cuda(device)  # 移动到GPU

    # 使用 DataLoader 进行批量推理
    dataset = TensorDataset(regions_batch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            batch_regions = batch[0]  # 批量区域
            if isinstance(model, open_clip.model.CLIP):
                region_features_batch = model.encode_image(batch_regions)
            else:
                region_features_batch = model.get_image_features(batch_regions)    # SigLIP
            region_features.append(region_features_batch)

    region_features = torch.cat(region_features, dim=0)  # [num_boxes, feature_dim]
    region_features = F.normalize(region_features, p=2, dim=1)  # 对每行（特征向量）进行 L2 归一化

    # Step 3: 计算 Region Proposal 与 GPT 预测类别的相似度
    region_features, text_features = region_features.float(), text_features.float()  # 转为 float32
    if isinstance(model, open_clip.model.CLIP):
        similarity_scores = torch.matmul(region_features, text_features.T)  # [num_boxes, num_gpt_classes]
    else:
        similarity_scores = torch.matmul(region_features, text_features.T) * model.logit_scale.exp() + model.logit_bias  # [num_boxes, num_gpt_classes]
        similarity_scores = torch.sigmoid(similarity_scores)

    # 如果 full_categories 不为空，则将 similarity_scores 中 gpt_predicted_classes 以外的类别所在列删掉
    if full_categories:
        # 获取 gpt_predicted_classes 在 full_categories 中的索引
        indices = [full_categories.index(cls) for cls in gpt_predicted_classes if cls in full_categories]
        # 使用这些索引选择 similarity_scores 中的列
        similarity_scores = similarity_scores[:, indices]

    similarity_labels = []
    similarity_scores_max = []  # 用于存储每个 box 的新分数（拼接后的字符串）
    
    for i in range(similarity_scores.shape[0]):  # 遍历每个 box
        # 获取当前 box 对应的最大相似度的类别索引
        max_sim_index = similarity_scores[i].argmax().item()
        # 根据最大相似度的索引分配标签
        similarity_labels.append(gpt_predicted_classes[max_sim_index][0])
        # 获取 similarity_scores 中的最大值作为分数
        max_sim_score = similarity_scores[i].max().item()
        similarity_scores_max.append(max_sim_score)
    if show_similarities:
        visualize_prediction(np.array(image), boxes, similarity_labels, scores=similarity_scores_max, dpi=100, title='Similarity Scores')
    
    start_time = time.time()
    # Step 4: Extract final boxes and labels based on assignments
    assignments = assign_labels_with_constraints(similarity_scores, gpt_predicted_counts)
    end_time = time.time()
    # print(f"Time taken for assignment: {end_time - start_time} seconds")
    if not assignments:
        print("Warning: No assignments found. Returning empty results.")
        return [], [], [], []  # 返回空结果

    boxes_final, labels_final, segmentations_final, scores_final = [], [], [], []

    for region_idx, label_idx in assignments:
        boxes_final.append(boxes[region_idx])  # Get the bounding box for this region
        labels_final.append(gpt_predicted_classes[label_idx])  # Get the label name for this assignment
        if segmentations is not None:
            segmentations_final.append(segmentations[region_idx])
        scores_final.append(similarity_scores_max[region_idx])
        
    return boxes_final, labels_final, segmentations_final, scores_final


def assign_labels_by_similarity(image, boxes, object_cnt, text_features,
                           model, preprocess, segmentations=None,
                           crop_scale=1.2, full_categories=None, min_crop_width=0, 
                           show_similarities=False):
    """for ablation study

    Args:
        image (np.array): img_array
        boxes (_type_): _description_
        object_cnt (_type_): _description_
        text_features (_type_): _description_
        model (_type_): _description_
        preprocess (_type_): _description_
        crop_scale (float, optional): _description_. Defaults to 1.2.
        min_crop_width (int, optional): _description_. Defaults to 0.
        show_similarities (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    device = next(model.parameters()).device
    # Step 1: 裁剪 Region Proposal 区域并计算图像特征
    region_features = []
    regions = []
    # 获取图像的宽和高
    image_width, image_height = image.size

    for box in boxes:
        x, y, w, h = box

        if x > image_width or y > image_height:
            print(f'Warning: box {box} out of image size {image.size}')
            x = image_width - 2
            y = image_height - 2
            print("---------------")
        
        # 计算裁剪框的中心点
        center_x = x + w / 2
        center_y = y + h / 2

        # 计算裁剪框的长宽（根据 crop_scale 调整） the 5 min avg bbox areas range from 12 (whistle) to 94 (baterry) 
        new_w = max(min_crop_width, w * crop_scale)
        new_h = max(min_crop_width, h * crop_scale)

        # 计算裁剪框的左上角和右下角坐标
        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2


        # 确保裁剪框不会超出图像边界
        new_x1 = max(0, new_x1)  # 左边界
        new_y1 = max(0, new_y1)  # 上边界
        new_x2 = min(image_width, new_x2)  # 右边界
        new_y2 = min(image_height, new_y2)  # 下边界
            
        region = image.crop((new_x1, new_y1, new_x2, new_y2))
        regions.append(region)
    
    if isinstance(model, open_clip.model.CLIP):
        regions = [preprocess(region).unsqueeze(0) for region in regions]
    else:
        regions = [preprocess(region) for region in regions]
    
    # 将所有 region 拼接成一个 batch
    regions_batch = torch.cat(regions, dim=0).cuda(device)  # 移动到GPU

    # 使用 DataLoader 进行批量推理
    dataset = TensorDataset(regions_batch)
    batch_size = 200  # 初始批量大小
    max_attempts = 5  # 最大尝试次数
    
    for attempt in range(max_attempts):
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for batch in dataloader:
                    batch_regions = batch[0]  # 批量区域
                    if isinstance(model, open_clip.model.CLIP):
                        region_features_batch = model.encode_image(batch_regions)
                    else:
                        region_features_batch = model(batch_regions)['pooler_output']    # SigLIP
                    region_features.append(region_features_batch)
            break  # 如果成功执行，跳出循环
        except RuntimeError as e:
            if "out of memory" in str(e) and attempt < max_attempts - 1:
                # 清空显存
                torch.cuda.empty_cache()
                # 减小批量大小
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size to {batch_size} due to OOM error")
                # 清空之前的结果
                region_features = []
                continue
            else:
                raise e  # 如果尝试次数用完或不是OOM错误，则抛出异常

    region_features = torch.cat(region_features, dim=0)  # [num_boxes, feature_dim]
    region_features = F.normalize(region_features, p=2, dim=1)  # 对每行（特征向量）进行 L2 归一化

    # Step 2: 提取 GPT 预测的目标类别及其计数
    gpt_predicted_classes = [cls for cls, count in object_cnt.items() if count > 0]
    gpt_predicted_counts = {cls: count for cls, count in object_cnt.items() if count > 0}
    # print(gpt_predicted_counts)
    
    if not gpt_predicted_classes:
        print("Warning: gpt_predicted_classes is empty. Returning empty results.")
        return [], [], [], []  # 返回空结果

    # Step 3: 计算 Region Proposal 与 GPT 预测类别的相似度
    region_features, text_features = region_features.float(), text_features.float()  # 转为 float32
    similarity_scores = torch.matmul(region_features, text_features.T)  # [num_boxes, num_gpt_classes]
    
    # 如果 full_categories 不为空，则将 similarity_scores 中 gpt_predicted_classes 以外的类别所在列删掉
    if full_categories is not None:
        # 获取 gpt_predicted_classes 在 full_categories 中的索引
        indices = [full_categories.index(cls) for cls in gpt_predicted_classes if cls in full_categories]
        # 使用这些索引选择 similarity_scores 中的列
        similarity_scores = similarity_scores[:, indices]

    similarity_labels = []
    similarity_scores_max = []  # 用于存储每个 box 的新分数（拼接后的字符串）
    labels_final = []
    for i in range(similarity_scores.shape[0]):  # 遍历每个 box
        # 获取当前 box 对应的最大相似度的类别索引
        max_sim_index = similarity_scores[i].argmax().item()
        # 根据最大相似度的索引分配标签
        similarity_labels.append(gpt_predicted_classes[max_sim_index][0])
        labels_final.append(gpt_predicted_classes[max_sim_index])
        # 获取 similarity_scores 中的最大值作为分数
        max_sim_score = similarity_scores[i].max().item()
        # print(max_sim_score)
        similarity_scores_max.append(max_sim_score)
    if show_similarities:
        visualize_prediction(np.array(image), boxes, similarity_labels, scores=similarity_scores_max, dpi=100, title='Similarity Scores')

        
    return boxes, labels_final, segmentations, similarity_scores_max