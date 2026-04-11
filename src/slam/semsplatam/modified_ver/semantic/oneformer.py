import torch
import torch.nn.functional as F
from src.visualization.o3d_utils import normalized

def positive_normalize(logits, dim=-1, min=0.):  # logits: [..., num_classes]
    mask = (logits > min)
    sum_positive = torch.sum(logits * mask, dim=dim, keepdim=True)
    safe_sum = torch.where(sum_positive == 0, torch.tensor(1.0, device=logits.device), sum_positive)
    normalized_logits = torch.where(mask, logits / safe_sum, torch.tensor(0.0, device=logits.device))
    return normalized_logits

def oneformer_segmentation(image, oneformer_processor, oneformer_model, rank, num_classes=102):
    inputs = oneformer_processor(images=image, task_inputs=["semantic"], return_tensors="pt").to(rank)
    outputs = oneformer_model(**inputs)
    oneformer_processor.do_reduce_labels = True

    target_sizes = [image.size[::-1]]

    class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
    masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

    # Remove the null class `[..., :-1]`
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

    # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    batch_size = class_queries_logits.shape[0]

    # Resize logits and compute semantic segmentation maps
    if target_sizes is not None:
        if batch_size != len(target_sizes):
            raise ValueError(
                "Make sure that you pass in as many target sizes as the batch dimension of the logits"
            )

        semantic_segmentation = []
        semantic_logits = []
        for idx in range(batch_size):
            resized_logits = torch.nn.functional.interpolate(
                segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
            )
            semantic_map = resized_logits[0].argmax(dim=0)
            semantic_map[semantic_map>num_classes-1]=0
            semantic_segmentation.append(semantic_map)

            logits = resized_logits[0].permute(1, 2, 0)
            logits = logits[...,:num_classes]
            normalized_logits = positive_normalize(logits, dim=-1, min=0.0)
            semantic_logits.append(normalized_logits)
    return semantic_segmentation,semantic_logits