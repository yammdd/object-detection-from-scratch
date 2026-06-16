import torch


def compute_iou_matrix(boxes_a, boxes_b):
    # Intersection
    x0 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))
    y0 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    x1 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    y1 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter = (x1 - x0).clamp(min=0) * (y1 - y0).clamp(min=0)

    # Areas
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter + 1e-7

    return inter / union


def nms_single_class(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        # Pick the best
        idx = order[0]
        keep.append(idx)

        if order.numel() == 1:
            break

        # Compute IoU of the picked box against all remaining
        remaining = order[1:]
        ious = compute_iou_matrix(
            boxes[idx].unsqueeze(0),   # (1, 4)
            boxes[remaining]           # (K, 4)
        ).squeeze(0)                   # (K,)

        # Keep only boxes with IoU <= threshold
        mask = ious <= iou_threshold
        order = remaining[mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def nms_per_class(boxes, scores, class_ids, num_classes,
                  iou_threshold=0.5, max_per_class=100):
    kept_boxes = []
    kept_scores = []
    kept_classes = []

    for cls_id in range(num_classes):
        cls_mask = class_ids == cls_id
        if cls_mask.sum() == 0:
            continue

        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        # Apply NMS
        keep = nms_single_class(cls_boxes, cls_scores, iou_threshold)

        # Limit detections per class
        if keep.numel() > max_per_class:
            keep = keep[:max_per_class]

        kept_boxes.append(cls_boxes[keep])
        kept_scores.append(cls_scores[keep])
        kept_classes.append(torch.full((keep.numel(),), cls_id,
                                       dtype=torch.long,
                                       device=boxes.device))

    if len(kept_boxes) > 0:
        kept_boxes = torch.cat(kept_boxes, dim=0)
        kept_scores = torch.cat(kept_scores, dim=0)
        kept_classes = torch.cat(kept_classes, dim=0)
    else:
        kept_boxes = torch.zeros((0, 4), device=boxes.device)
        kept_scores = torch.zeros((0,), device=boxes.device)
        kept_classes = torch.zeros((0,), dtype=torch.long, device=boxes.device)

    return kept_boxes, kept_scores, kept_classes


def post_process_single_image(boxes, scores, class_ids, num_classes,
                              conf_threshold=0.3, iou_threshold=0.5,
                              max_detections=300):
    # 1. Confidence threshold
    conf_mask = scores > conf_threshold
    boxes = boxes[conf_mask]
    scores = scores[conf_mask]
    class_ids = class_ids[conf_mask]

    if boxes.numel() == 0:
        return (torch.zeros((0, 4), device=boxes.device),
                torch.zeros((0,), device=boxes.device),
                torch.zeros((0,), dtype=torch.long, device=boxes.device))

    # 2. Per-class NMS
    det_boxes, det_scores, det_classes = nms_per_class(
        boxes, scores, class_ids, num_classes,
        iou_threshold=iou_threshold
    )

    # 3. Top-K by score
    if det_boxes.shape[0] > max_detections:
        topk = det_scores.argsort(descending=True)[:max_detections]
        det_boxes = det_boxes[topk]
        det_scores = det_scores[topk]
        det_classes = det_classes[topk]

    return det_boxes, det_scores, det_classes