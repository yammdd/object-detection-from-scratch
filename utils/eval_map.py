import torch
import numpy as np

from models.fcos_detector import FPN_STRIDES
from utils.decoder import decode_fcos_outputs
from utils.nms import post_process_single_image


def _compute_iou_matrix_np(boxes_a, boxes_b):
    x0 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)  # (N, M)
    y0 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x1 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y1 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)

    inter = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - inter + 1e-7

    return inter / union


def _compute_ap_101(recalls, precisions):
    if len(recalls) == 0:
        return 0.0

    # Prepend (0, 1) and append (1, 0) sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for rp in recall_points:
        # Find the smallest recall in mrec that is >= rp
        idx = np.searchsorted(mrec, rp, side='left')
        if idx < len(mpre):
            ap += mpre[idx]

    ap /= 101.0
    return ap


def _match_predictions_to_gt(pred_boxes, pred_scores, pred_classes,
                              gt_boxes, gt_classes, num_classes,
                              iou_threshold):
    class_matches = {c: [] for c in range(num_classes)}
    class_num_gt = {c: 0 for c in range(num_classes)}

    # Count GT per class
    for c in range(num_classes):
        class_num_gt[c] = int((gt_classes == c).sum())

    if len(pred_boxes) == 0:
        return class_matches, class_num_gt

    if len(gt_boxes) == 0:
        # All predictions are FP
        for i in range(len(pred_boxes)):
            c = int(pred_classes[i])
            class_matches[c].append((float(pred_scores[i]), False))
        return class_matches, class_num_gt

    # Compute IoU matrix between predictions and GT
    iou_matrix = _compute_iou_matrix_np(pred_boxes, gt_boxes)  # (D, G)

    # Track which GT boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)

    # Sort predictions by score descending (greedy matching)
    sorted_idx = np.argsort(-pred_scores)

    for di in sorted_idx:
        pc = int(pred_classes[di])
        score = float(pred_scores[di])

        # Find GT boxes of the same class
        gt_same_class = np.where(gt_classes == pc)[0]

        if len(gt_same_class) == 0:
            class_matches[pc].append((score, False))
            continue

        # IoU of this prediction with all same-class GT
        ious = iou_matrix[di, gt_same_class]

        # Find best unmatched GT
        best_local_idx = np.argmax(ious)
        best_iou = ious[best_local_idx]
        best_gt_idx = gt_same_class[best_local_idx]

        if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
            class_matches[pc].append((score, True))
            gt_matched[best_gt_idx] = True
        else:
            class_matches[pc].append((score, False))

    return class_matches, class_num_gt


@torch.no_grad()
def evaluate_map(model, dataloader, device, num_classes=5,
                 conf_threshold=0.01, iou_nms=0.5, max_detections=300):
    model.eval()

    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.5, 0.55, ..., 0.95]

    all_class_matches = {
        iou_t: {c: [] for c in range(num_classes)}
        for iou_t in iou_thresholds
    }
    all_class_num_gt = {c: 0 for c in range(num_classes)}

    for images, targets, metas in dataloader:
        images = images.to(device)
        B = images.shape[0]

        # Forward pass
        cls_outputs, reg_outputs, ctr_outputs = model(images)

        # Decode all FPN levels
        all_boxes, all_scores, all_classes, _ = decode_fcos_outputs(
            cls_outputs, reg_outputs, ctr_outputs, FPN_STRIDES
        )

        # Process each image in the batch
        for b in range(B):
            det_boxes, det_scores, det_classes = post_process_single_image(
                all_boxes[b], all_scores[b], all_classes[b],
                num_classes=num_classes,
                conf_threshold=conf_threshold,
                iou_threshold=iou_nms,
                max_detections=max_detections,
            )

            pred_boxes_np = det_boxes.cpu().numpy()
            pred_scores_np = det_scores.cpu().numpy()
            pred_classes_np = det_classes.cpu().numpy()

            gt_tensor = targets[b]  # (N, 5) [class_id, x0, y0, x1, y1]
            if gt_tensor.numel() > 0 and gt_tensor.shape[0] > 0:
                gt_classes_np = gt_tensor[:, 0].cpu().numpy().astype(np.int64)
                gt_boxes_np = gt_tensor[:, 1:5].cpu().numpy()
            else:
                gt_classes_np = np.zeros((0,), dtype=np.int64)
                gt_boxes_np = np.zeros((0, 4), dtype=np.float32)

            # Accumulate GT counts
            for c in range(num_classes):
                all_class_num_gt[c] += int((gt_classes_np == c).sum())

            # Match at each IoU threshold
            for iou_t in iou_thresholds:
                class_matches, _ = _match_predictions_to_gt(
                    pred_boxes_np, pred_scores_np, pred_classes_np,
                    gt_boxes_np, gt_classes_np, num_classes, iou_t
                )
                for c in range(num_classes):
                    all_class_matches[iou_t][c].extend(class_matches[c])

    ap_per_class_per_iou = {}  # (iou_t, class_id) -> AP

    for iou_t in iou_thresholds:
        for c in range(num_classes):
            matches = all_class_matches[iou_t][c]
            num_gt = all_class_num_gt[c]

            if num_gt == 0:
                # No GT for this class, AP is 0
                ap_per_class_per_iou[(iou_t, c)] = 0.0
                continue

            if len(matches) == 0:
                ap_per_class_per_iou[(iou_t, c)] = 0.0
                continue

            # Sort by score descending
            matches.sort(key=lambda x: x[0], reverse=True)
            scores_sorted = np.array([m[0] for m in matches])
            tp_sorted = np.array([m[1] for m in matches], dtype=np.float64)

            # Cumulative TP and FP
            cum_tp = np.cumsum(tp_sorted)
            cum_fp = np.cumsum(1 - tp_sorted)

            # Precision and recall
            precisions = cum_tp / (cum_tp + cum_fp + 1e-7)
            recalls = cum_tp / num_gt

            # Compute AP
            ap = _compute_ap_101(recalls, precisions)
            ap_per_class_per_iou[(iou_t, c)] = ap

    # mAP@0.5
    ap_50_values = [ap_per_class_per_iou[(0.5, c)] for c in range(num_classes)
                    if all_class_num_gt[c] > 0]
    mAP_50 = float(np.mean(ap_50_values)) if ap_50_values else 0.0

    # mAP@0.5:0.95
    ap_all_values = []
    for iou_t in iou_thresholds:
        for c in range(num_classes):
            if all_class_num_gt[c] > 0:
                ap_all_values.append(ap_per_class_per_iou[(iou_t, c)])
    mAP_50_95 = float(np.mean(ap_all_values)) if ap_all_values else 0.0

    # Per-class AP@0.5
    per_class_AP50 = {}
    for c in range(num_classes):
        per_class_AP50[c] = ap_per_class_per_iou.get((0.5, c), 0.0)

    model.train()

    return {
        'mAP_50': mAP_50,
        'mAP_50_95': mAP_50_95,
        'per_class_AP50': per_class_AP50,
    }