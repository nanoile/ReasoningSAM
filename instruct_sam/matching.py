import numpy as np
import torch
from instruct_sam.visualize import visualize_prediction
# from instruct_sam.visualize import CATEGORIES
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torchvision import transforms
import pulp
import open_clip
import time


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
    Args:
        clip_model: 'dfn2b', 'georsclip', 'georsclip-b32', 'remoteclip', 'remoteclip-b32', 'skyclip', 'skyclip-b32'
        ckpt_path: The path to the checkpoint file
    Returns:
        model, tokenizer, preprocess
    """
    if ckpt_path is None:
        raise ValueError(f"ckpt_path is None for {clip_model}")
    if clip_model == 'dfn2b':
        # ViT-L/14 (DFN)
        model_name = 'ViT-L-14'  # ('ViT-L-14', 'dfn2b_s39b'),
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt_path)
        tokenizer = open_clip.get_tokenizer(model_name)

    elif clip_model == 'remoteclip':
        model_name = 'ViT-L-14'
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        message = model.load_state_dict(ckpt)
        tokenizer = open_clip.get_tokenizer(model_name)

    elif clip_model == 'remoteclip-b32':
        model_name = 'ViT-B-32'
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=ckpt_path)
        tokenizer = open_clip.get_tokenizer(model_name)

    elif clip_model == 'georsclip':
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L/14", pretrained=ckpt_path)
        preprocess = get_preprocess_rs5m()
        tokenizer = open_clip.tokenize

    elif clip_model == 'georsclip-b32':
        # GeoRSCLIP
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B/32", pretrained=ckpt_path)
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
        ThirdPartyClip.__bases__ = (
            open_clip.model.CLIP,) + ThirdPartyClip.__bases__

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
        ThirdPartyClip.__bases__ = (
            open_clip.model.CLIP,) + ThirdPartyClip.__bases__

    else:
        raise ValueError(f"Invalid clip model: {clip_model}")

    model.cuda(device).eval()
    return model, tokenizer, preprocess


def filter_boxes(boxes, segmentations, scores, labels, threshold):
    """
    Filter the bounding boxes based on the score threshold. If the number of boxes left after filtering is less than min_box, keep the min_box boxes with the highest confidence.
    Args:
        boxes (list): The bounding box coordinates, shape is [num_boxes, 4]
        scores (list): The bounding box scores, shape is [num_boxes]
        labels (list): The bounding box labels, shape is [num_boxes]
        threshold (float): The score threshold
    Returns:
        The filtered bounding box coordinates, scores and labels
    """
    # Convert all inputs to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    labels = np.array(labels)

    if segmentations is None:
        segmentations = boxes
    else:
        # Keep segmentations as a list, because each segmentation mask may have a different number of points
        segmentations = list(segmentations)

    # Use boolean indexing to filter
    mask = scores >= threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]
    filtered_segmentations = [segmentations[i]
                              for i in range(len(segmentations)) if mask[i]]

    return filtered_boxes.tolist(), filtered_scores.tolist(), filtered_labels.tolist(), filtered_segmentations


def assign_labels_with_constraints(similarity_scores, gpt_predicted_counts):
    """
    Assign region proposals to labels based on similarity scores and constraints.

    Args:
        similarity_scores (np.ndarray): np.ndarray of shape (N, M), similarity scores between regions and labels.
        gpt_predicted_counts (dict): number of regions to assign for each label (e.g., {"harbor": 1, "ship": 4}).

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
    prob += pulp.lpSum(cost_matrix[i, j] * X[i, j]
                       for i in range(num_regions) for j in range(num_labels))

    # Step 4: Add constraints
    # Constraint 1: Each region proposal can be assigned to at most one label
    for i in range(num_regions):
        prob += pulp.lpSum(X[i, j] for j in range(num_labels)) <= 1

    # Constraint 2: Each label must be assigned to exactly the specified number of regions
    total_gpt_count = sum(gpt_predicted_counts.values())
    if num_regions <= total_gpt_count:
        # Case 1: Total regions are fewer than total GPT-predicted counts
        prob += pulp.lpSum(X[i, j] for i in range(num_regions)
                           for j in range(num_labels)) == num_regions
        label_indices = list(gpt_predicted_counts.keys())
        for j, label in enumerate(label_indices):
            prob += pulp.lpSum(X[i, j] for i in range(num_regions)
                               ) <= gpt_predicted_counts[label]
    else:
        # Case 2: Use the original constraint for each label
        label_indices = list(gpt_predicted_counts.keys())
        for j, label in enumerate(label_indices):
            prob += pulp.lpSum(X[i, j] for i in range(num_regions)
                               ) == gpt_predicted_counts[label]

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
                           crop_scale=1.2, batch_size=200, min_crop_width=0,
                           show_similarities=False, zero_count_warning=True):
    """
    Assigns region proposals (bounding boxes) to categories based on similarity scores and object count constraints.

    This function extracts features from each region proposal, computes similarity scores with the provided text features,
    and solves an assignment problem to match regions to categories according to the predicted object counts.

    Args:
        image (PIL.Image or np.ndarray): The input image.
        boxes (list): List of bounding boxes, each in the format [x, y, w, h].
        object_cnt (dict): Dictionary mapping category names to predicted object counts (e.g., {"car": 3, "ship": 2}).
        text_features (torch.Tensor): Text feature embeddings for each category, shape [num_categories, feature_dim].
        model (torch.nn.Module): The CLIP or compatible vision-language model.
        preprocess (callable): Preprocessing function for image regions.
        segmentations (list, optional): List of segmentation masks for each region. Defaults to None.
        crop_scale (float, optional): Scale factor for cropping regions around each box. Defaults to 1.2.
        batch_size (int, optional): Batch size for feature extraction. Defaults to 200.
        min_crop_width (int, optional): Minimum width/height for cropped regions. Defaults to 0.
        show_similarities (bool, optional): If True, visualize similarity scores for each region. Defaults to False.
        zero_count_warning (bool, optional): If True, print a warning when no valid categories are predicted. Defaults to True.

    Returns:
        boxes_final (list): List of matched bounding boxes.
        labels_final (list): List of assigned category labels for each box.
        segmentations_final (list): List of matched segmentations (if provided).
        scores_final (list): List of similarity scores for each matched box.

    Note:
        - If no valid categories are predicted (i.e., all counts are zero), returns empty lists.
    """
    device = next(model.parameters()).device

    # Step 1: Extract the GPT predicted target classes and counts
    gpt_predicted_classes = [cls for cls,
                             count in object_cnt.items() if count > 0]
    gpt_predicted_counts = {cls: count for cls,
                            count in object_cnt.items() if count > 0}
    # print(gpt_predicted_counts)

    if not gpt_predicted_classes:
        if zero_count_warning:
            print("Warning: gpt_predicted_classes is empty. Returning empty results.")
        return [], [], [], []  # Return empty results

    # Step 2: Crop the Region Proposal and calculate the image features
    region_features = []
    regions = []
    # Get the width and height of the image
    image_width, image_height = image.size

    for box in boxes:
        x, y, w, h = box

        if x > image_width or y > image_height:
            print(f'Warning: box {box} out of image size {image.size}')
            x = image_width - 2
            y = image_height - 2
            print("---------------")

        # Calculate the center point of the cropping box
        center_x = x + w / 2
        center_y = y + h / 2

        # Calculate the length and width of the cropping box (adjust according to crop_scale)
        new_w = max(min_crop_width, w * crop_scale)
        new_h = max(min_crop_width, h * crop_scale)

        # Calculate the coordinates of the top-left and bottom-right corners of the cropping box
        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2

        # Ensure the cropping box does not exceed the image boundaries
        new_x1 = max(0, new_x1)  # Left boundary
        new_y1 = max(0, new_y1)  # Top boundary
        new_x2 = min(image_width, new_x2)  # Right boundary
        new_y2 = min(image_height, new_y2)  # Bottom boundary

        region = image.crop((new_x1, new_y1, new_x2, new_y2))
        if isinstance(model, open_clip.model.CLIP):
            regions.append(preprocess(region).unsqueeze(0))
        else:
            regions.append(preprocess(region))

    # Concatenate all regions into a batch
    regions_batch = torch.cat(regions, dim=0).cuda(device)  # Move to GPU

    # Use DataLoader for batch inference
    dataset = TensorDataset(regions_batch)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            batch_regions = batch[0]  # Batch regions
            if isinstance(model, open_clip.model.CLIP):
                region_features_batch = model.encode_image(batch_regions)
            else:
                region_features_batch = model.get_image_features(
                    batch_regions)    # SigLIP
            region_features.append(region_features_batch)

    # [num_boxes, feature_dim]
    region_features = torch.cat(region_features, dim=0)
    # Normalize each row (feature vector) to L2
    region_features = F.normalize(region_features, p=2, dim=1)

    # Step 3: Calculate the similarity between Region Proposal and GPT predicted classes
    region_features, text_features = region_features.float(
    ), text_features.float()  # Convert to float32
    if isinstance(model, open_clip.model.CLIP):
        # [num_boxes, num_gpt_classes]
        similarity_scores = torch.matmul(region_features, text_features.T)
    else:
        similarity_scores = torch.matmul(region_features, text_features.T) * \
            model.logit_scale.exp() + \
            model.logit_bias  # [num_boxes, num_gpt_classes]
        similarity_scores = torch.sigmoid(similarity_scores)

    similarity_labels = []
    # Store the new scores for each box (concatenated string)
    similarity_scores_max = []

    for i in range(similarity_scores.shape[0]):  # Iterate over each box
        # Get the index of the box with the highest similarity
        max_sim_index = similarity_scores[i].argmax().item()
        # Assign the label based on the highest similarity index
        similarity_labels.append(gpt_predicted_classes[max_sim_index][0])
        # Get the maximum value of similarity_scores as the score
        max_sim_score = similarity_scores[i].max().item()
        similarity_scores_max.append(max_sim_score)
    if show_similarities:
        visualize_prediction(np.array(image), boxes, similarity_labels,
                             scores=similarity_scores_max, dpi=100, title='Similarity Scores')

    # start_time = time.time()
    # Step 4: Extract final boxes and labels based on assignments
    assignments = assign_labels_with_constraints(
        similarity_scores, gpt_predicted_counts)
    # end_time = time.time()
    # print(f"Time taken for assignment: {end_time - start_time} seconds")
    if not assignments:
        print("Warning: No assignments found. Returning empty results.")
        return [], [], [], []  # Return empty results

    boxes_final, labels_final, segmentations_final, scores_final = [], [], [], []

    for region_idx, label_idx in assignments:
        # Get the bounding box for this region
        boxes_final.append(boxes[region_idx])
        # Get the label name for this assignment
        labels_final.append(gpt_predicted_classes[label_idx])
        if segmentations is not None:
            segmentations_final.append(segmentations[region_idx])
        scores_final.append(similarity_scores_max[region_idx])

    return boxes_final, labels_final, segmentations_final, scores_final
