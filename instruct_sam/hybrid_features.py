import math
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from pycocotools import mask as mask_utils


CLIP_PIXEL_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)


def _ensure_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def decode_segmentation(
    segmentation: Sequence,
    height: int,
    width: int,
) -> np.ndarray:
    """
    Decode a COCO-format segmentation into a binary mask.

    Args:
        segmentation: Polygon list, uncompressed RLE dict, numpy array mask, or bool array.
        height: Target mask height.
        width: Target mask width.

    Returns:
        np.ndarray: Binary mask of shape (height, width) with dtype=np.uint8 and values {0, 1}.
    """
    if segmentation is None:
        raise ValueError("Segmentation is None while decoding mask")

    if isinstance(segmentation, np.ndarray):
        mask = segmentation.astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != (height, width):
            raise ValueError(
                f"Segmentation mask shape {mask.shape} does not match image size {(height, width)}"
            )
        return (mask > 0).astype(np.uint8)

    if isinstance(segmentation, dict):
        rle = mask_utils.frPyObjects([segmentation], height, width)[0] if "counts" not in segmentation else segmentation
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return (mask > 0).astype(np.uint8)

    if isinstance(segmentation, (list, tuple)):
        rles = mask_utils.frPyObjects(segmentation, height, width)
        mask = mask_utils.decode(rles)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        return mask.astype(np.uint8)

    raise TypeError(f"Unsupported segmentation type: {type(segmentation)}")


def _build_hybrid_views(
    image_np: np.ndarray,
    masks: Sequence[np.ndarray],
    blur_kernel: Tuple[int, int] = (15, 15),
) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Generate masked (local) and blurred (global) images following HybridGL.

    Args:
        image_np: H x W x 3 numpy array in uint8.
        masks: Sequence of H x W binary masks (dtype uint8 or bool).
        blur_kernel: Gaussian blur kernel size for background blending.

    Returns:
        A tuple (local_images, global_images) of PIL images.
    """
    if not masks:
        return [], []

    blurred_background = cv2.GaussianBlur(image_np, blur_kernel, 0)

    local_images: List[Image.Image] = []
    global_images: List[Image.Image] = []

    clip_mean_rgb = (CLIP_PIXEL_MEAN * 255.0).astype(np.float32)

    for mask in masks:
        mask_u8 = mask.astype(np.uint8)
        if mask_u8.shape != image_np.shape[:2]:
            raise ValueError(
                f"Mask shape {mask_u8.shape} does not match image size {image_np.shape[:2]}"
            )

        # Local image keeps the foreground sharp, background filled with CLIP mean colour.
        mask_expanded = mask_u8[..., None].astype(np.float32)
        local_arr = image_np.astype(np.float32) * mask_expanded + clip_mean_rgb * (1.0 - mask_expanded)
        local_img = Image.fromarray(local_arr.clip(0, 255).astype(np.uint8))
        local_images.append(local_img)

        # Global image blends sharp foreground with blurred background.
        mask_255 = mask_u8 * 255
        sharp_region = cv2.bitwise_and(image_np, image_np, mask=mask_255)
        inverted_mask = cv2.bitwise_not(mask_255)
        blurred_region = cv2.bitwise_and(blurred_background, blurred_background, mask=inverted_mask)
        global_arr = cv2.add(sharp_region, blurred_region)
        global_img = Image.fromarray(global_arr.astype(np.uint8))
        global_images.append(global_img)

    return local_images, global_images


def _apply_preprocess(
    images: Sequence[Image.Image],
    preprocess: transforms.Compose,
) -> torch.Tensor:
    if not images:
        return torch.empty(0)
    tensors = [preprocess(img) for img in images]
    return torch.stack(tensors, dim=0)


def compute_hybrid_region_features(
    image_pil: Image.Image,
    segmentations: Sequence,
    clip_model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: Optional[torch.device] = None,
    blur_kernel: Tuple[int, int] = (15, 15),
    fusion_weight: float = 0.5,
) -> torch.Tensor:
    """
    Compute HybridGL-style region features using CLIP image encoder.

    Args:
        image_pil: Source image in RGB.
        segmentations: Sequence of region proposals in COCO segmentation format.
        clip_model: CLIP image encoder with encode_image/get_image_features.
        preprocess: Transform that maps PIL image -> normalized tensor accepted by clip_model.
        device: Target torch device. Defaults to clip_model parameters device.
        blur_kernel: Gaussian blur kernel for global context view.
        fusion_weight: Weight for blending local/global features (0..1).

    Returns:
        torch.Tensor: Normalized features for each region, shape (num_regions, dim).
    """
    if device is None:
        device = next(clip_model.parameters()).device

    image_np = np.array(image_pil.convert("RGB"))
    height, width = image_np.shape[:2]

    masks: List[np.ndarray] = []
    for seg in segmentations:
        mask = decode_segmentation(seg, height, width)
        masks.append(mask)

    local_pil, global_pil = _build_hybrid_views(image_np, masks, blur_kernel=blur_kernel)

    if not local_pil:
        return torch.empty(0, device=device)

    local_batch = _apply_preprocess(local_pil, preprocess)
    global_batch = _apply_preprocess(global_pil, preprocess)

    local_batch = _ensure_device(local_batch, device)
    global_batch = _ensure_device(global_batch, device)

    with torch.no_grad():
        if hasattr(clip_model, "encode_image"):
            local_features = clip_model.encode_image(local_batch)
            global_features = clip_model.encode_image(global_batch)
        else:
            local_features = clip_model.get_image_features(local_batch)
            global_features = clip_model.get_image_features(global_batch)

    local_features = F.normalize(local_features.float(), p=2, dim=1)
    global_features = F.normalize(global_features.float(), p=2, dim=1)

    fusion_weight = float(np.clip(fusion_weight, 0.0, 1.0))
    if fusion_weight == 0.0:
        fused = global_features
    elif math.isclose(fusion_weight, 1.0):
        fused = local_features
    else:
        fused = fusion_weight * local_features + (1.0 - fusion_weight) * global_features

    fused = F.normalize(fused, p=2, dim=1)
    return fused
