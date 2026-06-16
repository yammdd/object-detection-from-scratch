import torch


def compute_centerness(reg_targets):
    l, t, r, b = reg_targets[:, 0], reg_targets[:, 1], \
                 reg_targets[:, 2], reg_targets[:, 3]

    lr = torch.min(l, r) / torch.max(l, r).clamp(min=1e-7)
    tb = torch.min(t, b) / torch.max(t, b).clamp(min=1e-7)

    centerness = torch.sqrt((lr * tb).clamp(min=0))
    return centerness


def assign_targets_to_points(
    points,          # (N_total, 2) - grid point centers (x, y) in image coords
    strides,         # (N_total,) - stride for each point
    gt_boxes,        # (M, 4) - [x0, y0, x1, y1] in image coords
    gt_classes,      # (M,) - class indices (0-indexed)
    scale_ranges,    # list of (lo, hi) per FPN level
    level_start_indices,  # list of int - start index for each level
    num_classes=5,
):
    N = points.shape[0]
    device = points.device

    # Initialize all as negative
    cls_targets = torch.full((N,), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros((N, 4), dtype=torch.float32, device=device)
    ctr_targets = torch.zeros((N,), dtype=torch.float32, device=device)

    if gt_boxes.shape[0] == 0:
        return cls_targets, reg_targets, ctr_targets

    M = gt_boxes.shape[0]

    # Compute (l, t, r, b) for ALL points against ALL GT boxes
    # points: (N, 2), gt_boxes: (M, 4)
    # Expand for broadcasting: points (N, 1, 2), gt_boxes (1, M, 4)
    x_c = points[:, 0:1]  # (N, 1)
    y_c = points[:, 1:2]  # (N, 1)

    gt_x0 = gt_boxes[:, 0].unsqueeze(0)  # (1, M)
    gt_y0 = gt_boxes[:, 1].unsqueeze(0)  # (1, M)
    gt_x1 = gt_boxes[:, 2].unsqueeze(0)  # (1, M)
    gt_y1 = gt_boxes[:, 3].unsqueeze(0)  # (1, M)

    l = x_c - gt_x0  # (N, M)
    t = y_c - gt_y0  # (N, M)
    r = gt_x1 - x_c  # (N, M)
    b = gt_y1 - y_c  # (N, M)

    reg_all = torch.stack([l, t, r, b], dim=2)  # (N, M, 4)

    # Condition 1: point must be inside the GT box (all of l,t,r,b > 0)
    inside_mask = reg_all.min(dim=2)[0] > 0  # (N, M)

    # Condition 2: max(l,t,r,b) must fall within the scale range of the
    # point's FPN level
    max_reg = reg_all.max(dim=2)[0]  # (N, M)

    # Build scale range tensor per point
    num_levels = len(scale_ranges)
    scale_lo = torch.zeros(N, dtype=torch.float32, device=device)
    scale_hi = torch.zeros(N, dtype=torch.float32, device=device)

    for lvl in range(num_levels):
        start = level_start_indices[lvl]
        end = level_start_indices[lvl + 1] if lvl + 1 < num_levels else N
        scale_lo[start:end] = scale_ranges[lvl][0]
        scale_hi[start:end] = scale_ranges[lvl][1]

    # Expand for broadcasting with M dimension
    scale_lo_exp = scale_lo.unsqueeze(1)  # (N, 1)
    scale_hi_exp = scale_hi.unsqueeze(1)  # (N, 1)

    scale_mask = (max_reg >= scale_lo_exp) & (max_reg < scale_hi_exp)  # (N, M)

    # Combined mask: inside GT AND correct scale
    valid_mask = inside_mask & scale_mask  # (N, M)

    gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
               (gt_boxes[:, 3] - gt_boxes[:, 1])  # (M,)

    # Set area to INF where not valid, then take argmin
    area_expanded = gt_areas.unsqueeze(0).expand(N, M)  # (N, M)
    area_masked = torch.where(valid_mask, area_expanded,
                              torch.tensor(float('inf'), device=device))

    # Find best GT for each point (smallest area)
    min_area, best_gt_idx = area_masked.min(dim=1)  # (N,), (N,)

    # Points that have at least one valid GT assignment
    positive_mask = min_area < float('inf')

    # Assign targets for positive points
    pos_indices = torch.where(positive_mask)[0]

    if pos_indices.numel() > 0:
        best_gt = best_gt_idx[pos_indices]

        cls_targets[pos_indices] = gt_classes[best_gt]

        # Gather regression targets
        reg_targets[pos_indices] = reg_all[pos_indices, best_gt]

        # Compute centerness
        ctr_targets[pos_indices] = compute_centerness(reg_targets[pos_indices])

    return cls_targets, reg_targets, ctr_targets


def assign_targets_batch(
    cls_outputs,     # list of (B, C, H_i, W_i)
    model,           # FCOSDetector - for strides, scale_ranges, compute_grid_points
    targets_list,    # list of Tensor (N_i, 5) - [class_id, x0, y0, x1, y1]
    num_classes=5,
):
    device = cls_outputs[0].device
    B = cls_outputs[0].shape[0]

    # Get feature map shapes
    feature_shapes = [(c.shape[2], c.shape[3]) for c in cls_outputs]

    # Compute grid points
    all_points, all_strides, level_starts = model.compute_grid_points(
        feature_shapes, device
    )

    N_total = all_points.shape[0]

    batch_cls = []
    batch_reg = []
    batch_ctr = []

    for b in range(B):
        targets = targets_list[b]  # (N_i, 5) [class_id, x0, y0, x1, y1]

        if targets.numel() == 0 or targets.shape[0] == 0:
            gt_boxes = torch.zeros((0, 4), device=device)
            gt_classes = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            gt_classes = targets[:, 0].long().to(device)
            gt_boxes = targets[:, 1:5].to(device)

        cls_t, reg_t, ctr_t = assign_targets_to_points(
            all_points, all_strides,
            gt_boxes, gt_classes,
            model.scale_ranges, level_starts,
            num_classes
        )

        batch_cls.append(cls_t)
        batch_reg.append(reg_t)
        batch_ctr.append(ctr_t)

    all_cls_targets = torch.stack(batch_cls, dim=0)  # (B, N_total)
    all_reg_targets = torch.stack(batch_reg, dim=0)  # (B, N_total, 4)
    all_ctr_targets = torch.stack(batch_ctr, dim=0)  # (B, N_total)

    return all_cls_targets, all_reg_targets, all_ctr_targets, \
           all_points, all_strides