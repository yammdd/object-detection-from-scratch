import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QualityFocalLoss(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_logits, targets_onehot, num_classes):
        pred_logits = pred_logits.float()
        targets_onehot = targets_onehot.float()

        # Sigmoid probability
        pred_sigmoid = torch.sigmoid(pred_logits)
        scale_factor = (targets_onehot - pred_sigmoid).abs().pow(self.beta)

        # Binary cross-entropy (per-element, per-class)
        bce = F.binary_cross_entropy_with_logits(
            pred_logits, targets_onehot, reduction='none'
        )

        loss = (scale_factor * bce).sum()
        return loss


class EIoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes, gt_boxes):
        pred_boxes = pred_boxes.float()
        gt_boxes = gt_boxes.float()

        # Extract coordinates
        pred_x0 = pred_boxes[:, 0]
        pred_y0 = pred_boxes[:, 1]
        pred_x1 = pred_boxes[:, 2]
        pred_y1 = pred_boxes[:, 3]

        gt_x0 = gt_boxes[:, 0]
        gt_y0 = gt_boxes[:, 1]
        gt_x1 = gt_boxes[:, 2]
        gt_y1 = gt_boxes[:, 3]

        # Areas
        pred_area = (pred_x1 - pred_x0).clamp(min=0) * \
                    (pred_y1 - pred_y0).clamp(min=0)
        gt_area = (gt_x1 - gt_x0).clamp(min=0) * \
                  (gt_y1 - gt_y0).clamp(min=0)

        # Intersection
        inter_x0 = torch.max(pred_x0, gt_x0)
        inter_y0 = torch.max(pred_y0, gt_y0)
        inter_x1 = torch.min(pred_x1, gt_x1)
        inter_y1 = torch.min(pred_y1, gt_y1)
        inter_area = (inter_x1 - inter_x0).clamp(min=0) * \
                     (inter_y1 - inter_y0).clamp(min=0)

        # Union
        union_area = pred_area + gt_area - inter_area + 1e-7

        # IoU
        iou = (inter_area / union_area).clamp(min=0, max=1.0)

        pred_cx = (pred_x0 + pred_x1) / 2
        pred_cy = (pred_y0 + pred_y1) / 2
        gt_cx = (gt_x0 + gt_x1) / 2
        gt_cy = (gt_y0 + gt_y1) / 2
        rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

        # Smallest enclosing box
        enc_x0 = torch.min(pred_x0, gt_x0)
        enc_y0 = torch.min(pred_y0, gt_y0)
        enc_x1 = torch.max(pred_x1, gt_x1)
        enc_y1 = torch.max(pred_y1, gt_y1)

        # Squared diagonal of enclosing box
        c2 = (enc_x1 - enc_x0) ** 2 + (enc_y1 - enc_y0) ** 2 + 1e-7

        # Clamp distance ratio to prevent Inf from degenerate boxes
        diou_penalty = (rho2 / c2).clamp(max=10.0)

        pred_w = (pred_x1 - pred_x0).clamp(min=1e-7)
        pred_h = (pred_y1 - pred_y0).clamp(min=1e-7)
        gt_w = (gt_x1 - gt_x0).clamp(min=1e-7)
        gt_h = (gt_y1 - gt_y0).clamp(min=1e-7)

        # Squared width and height of the enclosing box
        Cw2 = (enc_x1 - enc_x0) ** 2 + 1e-7
        Ch2 = (enc_y1 - enc_y0) ** 2 + 1e-7

        # Directly penalize width and height differences
        width_penalty = ((pred_w - gt_w) ** 2 / Cw2).clamp(max=10.0)
        height_penalty = ((pred_h - gt_h) ** 2 / Ch2).clamp(max=10.0)

        eiou = iou - diou_penalty - width_penalty - height_penalty

        # Clamp loss to [0, 4] to avoid gradient explosion from degenerate boxes
        loss = (1 - eiou).clamp(min=0, max=4.0).mean()
        return loss, iou.detach()  # Return IoU for QFL targets


class FCOSLoss(nn.Module):
    def __init__(self, num_classes=5, beta=2.0,
                 lambda_reg=1.0, lambda_ctr=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.qfl = QualityFocalLoss(beta=beta)
        self.eiou_loss = EIoULoss()
        self.lambda_reg = lambda_reg
        self.lambda_ctr = lambda_ctr

    def _decode_reg_to_boxes(self, reg_preds, points):
        x0 = points[:, 0] - reg_preds[:, 0]
        y0 = points[:, 1] - reg_preds[:, 1]
        x1 = points[:, 0] + reg_preds[:, 2]
        y1 = points[:, 1] + reg_preds[:, 3]
        return torch.stack([x0, y0, x1, y1], dim=1)

    def forward(self, cls_outputs, reg_outputs, ctr_outputs,
                cls_targets, reg_targets, ctr_targets,
                points):
        B = cls_outputs[0].shape[0]
        device = cls_outputs[0].device

        all_cls_pred = []
        all_reg_pred = []
        all_ctr_pred = []

        for cls, reg, ctr in zip(cls_outputs, reg_outputs, ctr_outputs):
            B_, C_, H_, W_ = cls.shape
            all_cls_pred.append(cls.permute(0, 2, 3, 1).reshape(B_, -1, C_))
            all_reg_pred.append(reg.permute(0, 2, 3, 1).reshape(B_, -1, 4))
            all_ctr_pred.append(ctr.permute(0, 2, 3, 1).reshape(B_, -1))

        cls_pred = torch.cat(all_cls_pred, dim=1)  # (B, N_total, C)
        reg_pred = torch.cat(all_reg_pred, dim=1)  # (B, N_total, 4)
        ctr_pred = torch.cat(all_ctr_pred, dim=1)  # (B, N_total)

        N_total = cls_pred.shape[1]

        cls_pred_flat = cls_pred.reshape(-1, self.num_classes)  # (B*N, C)
        reg_pred_flat = reg_pred.reshape(-1, 4)                 # (B*N, 4)
        ctr_pred_flat = ctr_pred.reshape(-1)                    # (B*N,)

        cls_targets_flat = cls_targets.reshape(-1)              # (B*N,)
        reg_targets_flat = reg_targets.reshape(-1, 4)           # (B*N, 4)
        ctr_targets_flat = ctr_targets.reshape(-1)              # (B*N,)

        # Expand points for batch
        points_flat = points.unsqueeze(0).expand(B, -1, -1).reshape(-1, 2)

        # Positive mask
        pos_mask = cls_targets_flat >= 0
        N_pos = pos_mask.sum().clamp(min=1).float()

        # Regression Loss (EIoU over positive points only)
        if pos_mask.any():
            # Decode predictions to boxes
            pred_boxes = self._decode_reg_to_boxes(
                reg_pred_flat[pos_mask], points_flat[pos_mask]
            )
            gt_boxes = self._decode_reg_to_boxes(
                reg_targets_flat[pos_mask], points_flat[pos_mask]
            )
            loss_reg, pos_iou = self.eiou_loss(pred_boxes, gt_boxes)
        else:
            loss_reg = torch.tensor(0.0, device=device)
            pos_iou = torch.zeros(0, device=device)

        # Classification Loss (QFL over ALL points)
        N_flat = cls_pred_flat.shape[0]
        qfl_targets = torch.zeros(
            N_flat, self.num_classes, dtype=torch.float32, device=device
        )
        if pos_mask.any():
            pos_indices = torch.where(pos_mask)[0]
            pos_classes = cls_targets_flat[pos_mask].long()
            iou_scores = pos_iou.detach().clamp(min=0, max=1.0)

            # Set soft targets: one_hot * IoU
            qfl_targets[pos_indices, pos_classes] = iou_scores

        loss_cls = self.qfl(cls_pred_flat, qfl_targets, self.num_classes) / N_pos

        # Centerness Loss (BCE over positive points only)
        if pos_mask.any():
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_pred_flat[pos_mask].float(),
                ctr_targets_flat[pos_mask].float(),
                reduction='sum'
            ) / N_pos
        else:
            loss_ctr = torch.tensor(0.0, device=device)

        # Total Loss
        total_loss = loss_cls + self.lambda_reg * loss_reg + \
                     self.lambda_ctr * loss_ctr

        loss_dict = {
            'loss_cls': loss_cls.item(),
            'loss_reg': loss_reg.item(),
            'loss_ctr': loss_ctr.item(),
            'total_loss': total_loss.item(),
            'N_pos': N_pos.item(),
        }

        if pos_iou.numel() > 0:
            loss_dict['mean_iou'] = pos_iou.mean().item()
        else:
            loss_dict['mean_iou'] = 0.0

        return total_loss, loss_dict