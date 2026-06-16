import torch


def decode_fcos_outputs(cls_outputs, reg_outputs, ctr_outputs, strides):
    device = cls_outputs[0].device
    B = cls_outputs[0].shape[0]
    num_classes = cls_outputs[0].shape[1]

    level_boxes = []
    level_scores = []
    level_classes = []
    level_cls_scores = []

    for level_idx, (cls, reg, ctr) in enumerate(
            zip(cls_outputs, reg_outputs, ctr_outputs)):

        stride = strides[level_idx]
        _, C, H, W = cls.shape

        # Generate grid center points for this level
        shifts_y = (torch.arange(0, H, dtype=torch.float32,
                                 device=device) + 0.5) * stride
        shifts_x = (torch.arange(0, W, dtype=torch.float32,
                                 device=device) + 0.5) * stride

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        # (H, W) each

        # Expand for batch: (1, H, W)
        shift_x = shift_x.unsqueeze(0)
        shift_y = shift_y.unsqueeze(0)

        # Decode boxes: (l, t, r, b) -> [x0, y0, x1, y1]
        # reg shape: (B, 4, H, W)
        x0 = shift_x - reg[:, 0, :, :]  # (B, H, W)
        y0 = shift_y - reg[:, 1, :, :]
        x1 = shift_x + reg[:, 2, :, :]
        y1 = shift_y + reg[:, 3, :, :]

        boxes = torch.stack([x0, y0, x1, y1], dim=-1)  # (B, H, W, 4)
        boxes = boxes.reshape(B, -1, 4)                 # (B, H*W, 4)

        # QFL scoring: sigmoid(cls) only
        cls_score = torch.sigmoid(cls)   # (B, C, H, W)

        # Reshape: (B, C, H, W) -> (B, H*W, C)
        cls_score_flat = cls_score.permute(0, 2, 3, 1).reshape(B, -1, C)

        # Best class per location
        max_scores, max_classes = cls_score_flat.max(dim=2)  # (B, H*W)

        level_boxes.append(boxes)
        level_scores.append(max_scores)
        level_classes.append(max_classes)
        level_cls_scores.append(cls_score_flat)

    # Concatenate all FPN levels
    all_boxes = torch.cat(level_boxes, dim=1)           # (B, N_total, 4)
    all_scores = torch.cat(level_scores, dim=1)         # (B, N_total)
    all_classes = torch.cat(level_classes, dim=1)        # (B, N_total)
    all_cls_scores = torch.cat(level_cls_scores, dim=1)  # (B, N_total, C)

    return all_boxes, all_scores, all_classes, all_cls_scores


def rescale_boxes(boxes, scale, pad_w, pad_h, orig_w, orig_h):
    boxes = boxes.clone().float()

    # Remove padding offset
    boxes[:, [0, 2]] -= pad_w
    boxes[:, [1, 3]] -= pad_h

    # Remove scaling
    boxes /= scale

    # Clip to original image boundaries
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

    # Filter degenerate boxes (too small)
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    valid_mask = (w > 1) & (h > 1)

    return boxes, valid_mask