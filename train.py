import os
import sys
import json
import time
import math
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcos_detector import FCOSDetector
from utils.dataset import DetectionDataset, collate_fn, CLASS_NAMES
from utils.augmentation import TrainAugmentation
from utils.target_assigner import assign_targets_batch
from utils.loss import FCOSLoss
from utils.eval_map import evaluate_map


def get_raw_model(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_lambda(warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warm-up from 0 to 1
            return max(epoch / warmup_epochs, 0.01)
        # Cosine decay from 1.0 down to 0.01
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * progress))
    return lr_lambda


def get_augment_stage(epoch, total_epochs):
    finetune_start = int(total_epochs * 0.8)
    advanced_start = int(total_epochs * 0.2)
    warmup_end = min(5, int(total_epochs * 0.05))

    if epoch < warmup_end:
        return 'warmup'
    elif epoch < advanced_start:
        return 'main'
    elif epoch < finetune_start:
        return 'advanced'
    else:
        return 'finetune'


def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    epoch, max_grad_norm=1.0):
    model.train()

    running_losses = {}
    num_batches = 0

    for batch_idx, (images, targets, metas) in enumerate(dataloader):
        images = images.to(device)

        # Forward pass
        cls_outputs, reg_outputs, ctr_outputs = model(images)

        # Assign targets
        raw_model = get_raw_model(model)
        cls_targets, reg_targets, ctr_targets, points, strides = \
            assign_targets_batch(cls_outputs, raw_model, targets,
                                 num_classes=raw_model.num_classes)

        # Compute loss
        total_loss, loss_dict = criterion(
            cls_outputs, reg_outputs, ctr_outputs,
            cls_targets, reg_targets, ctr_targets,
            points
        )

        # Check for NaN loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"  [WARNING] NaN/Inf loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping (prevents exploding gradients)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        # Accumulate losses
        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0) + v
        num_batches += 1

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            avg_total = running_losses.get('total_loss', 0) / num_batches
            avg_cls = running_losses.get('loss_cls', 0) / num_batches
            avg_reg = running_losses.get('loss_reg', 0) / num_batches
            avg_ctr = running_losses.get('loss_ctr', 0) / num_batches
            avg_iou = running_losses.get('mean_iou', 0) / num_batches
            n_pos = loss_dict.get('N_pos', 0)
            print(f"  [{batch_idx+1}/{len(dataloader)}] "
                  f"loss={avg_total:.4f} "
                  f"(cls={avg_cls:.4f} reg={avg_reg:.4f} ctr={avg_ctr:.4f}) "
                  f"IoU={avg_iou:.4f} N_pos={n_pos:.0f}")

    # Average losses
    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()

    running_losses = {}
    num_batches = 0

    for images, targets, metas in dataloader:
        images = images.to(device)

        cls_outputs, reg_outputs, ctr_outputs = model(images)

        raw_model = get_raw_model(model)
        cls_targets, reg_targets, ctr_targets, points, strides = \
            assign_targets_batch(cls_outputs, raw_model, targets,
                                 num_classes=raw_model.num_classes)

        total_loss, loss_dict = criterion(
            cls_outputs, reg_outputs, ctr_outputs,
            cls_targets, reg_targets, ctr_targets,
            points
        )

        for k, v in loss_dict.items():
            running_losses[k] = running_losses.get(k, 0) + v
        num_batches += 1

    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses


def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_mAP_50, best_mAP_50_95,
                    path, is_best=False):
    raw_model = get_raw_model(model)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'best_mAP_50': best_mAP_50,
        'best_mAP_50_95': best_mAP_50_95,
    }
    torch.save(checkpoint, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), 'best.pth')
        torch.save(checkpoint, best_path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model = get_raw_model(model)
    raw_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    best_mAP_50 = checkpoint.get('best_mAP_50', checkpoint.get('best_mAP', 0.0))
    best_mAP_50_95 = checkpoint.get('best_mAP_50_95', 0.0)
    
    return checkpoint['epoch'], best_val_loss, best_mAP_50, best_mAP_50_95


def main():
    parser = argparse.ArgumentParser(description='FCOS Object Detection Training')

    # Data paths
    parser.add_argument('--train_data', type=str,
                        default='./public/annotations/train.json',
                        help='Path to training annotation JSON')
    parser.add_argument('--val_data', type=str,
                        default='./public/annotations/val.json',
                        help='Path to validation annotation JSON')
    parser.add_argument('--image_dir', type=str,
                        default='./public/train/images',
                        help='Path to training images directory')
    parser.add_argument('--val_image_dir', type=str,
                        default='./public/val/images',
                        help='Path to validation images directory')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='./models/',
                        help='Directory to save checkpoints')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Base learning rate for head/FPN')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Number of warm-up epochs')

    # Loss hyperparameters
    parser.add_argument('--qfl_beta', type=float, default=2.0,
                        help='QFL focusing parameter')
    parser.add_argument('--lambda_reg', type=float, default=1.0,
                        help='Weight for regression loss')
    parser.add_argument('--lambda_ctr', type=float, default=0.5,
                        help='Weight for centerness loss (reduced, QFL handles quality)')

    # Training options
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='Validate every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0=disabled). '
                             'Stop if val loss does not improve for N '
                             'consecutive validation checks.')
    parser.add_argument('--early_stop_metric', type=str, default='mAP_50',
                        choices=['mAP_50', 'mAP_50_95', 'val_loss'],
                        help='Metric to use for early stopping and best checkpoint selection')

    args = parser.parse_args()

    # Set seeds
    set_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f"Device: {device}")
    if device.type == 'cuda':
        for gpu_i in range(num_gpus):
            print(f"  GPU {gpu_i}: {torch.cuda.get_device_name(gpu_i)} "
                  f"({torch.cuda.get_device_properties(gpu_i).total_memory / 1e9:.1f} GB)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("\nBuilding Datasets")

    # Training dataset
    train_dataset = DetectionDataset(
        ann_file=args.train_data,
        img_dir=args.image_dir,
        img_size=args.img_size,
        augment=None
    )

    # Create augmentation pipeline with dataset reference
    train_augment = TrainAugmentation(
        dataset=train_dataset,
        img_size=args.img_size,
        stage='warmup'
    )
    train_dataset.augment = train_augment

    # Validation dataset (no augmentation)
    val_dataset = DetectionDataset(
        ann_file=args.val_data,
        img_dir=args.val_image_dir,
        img_size=args.img_size,
        augment=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    print(f"Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} images, {len(val_loader)} batches")

    print("\nBuilding Model")
    model = FCOSDetector(num_classes=5, pretrained_backbone=True)
    model.to(device)
    model.count_parameters()

    param_groups = model.get_param_groups(args.lr, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    if num_gpus > 1:
        print(f"\n  Using DataParallel on {num_gpus} GPUs!")
        model = nn.DataParallel(model)
        print(f"  Per-GPU batch size: {args.batch_size // num_gpus}")

    # Learning rate scheduler
    lr_lambda = get_lr_lambda(args.warmup_epochs, args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Loss (QFL + EIoU)
    criterion = FCOSLoss(
        num_classes=5,
        beta=args.qfl_beta,
        lambda_reg=args.lambda_reg,
        lambda_ctr=args.lambda_ctr
    )

    start_epoch = 0
    best_val_loss = float('inf')
    best_mAP_50 = 0.0
    best_mAP_50_95 = 0.0
    epochs_no_improve = 0  # Early stopping counter

    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from: {args.resume}")
        start_epoch, best_val_loss, best_mAP_50, best_mAP_50_95 = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        start_epoch += 1  # Start from next epoch
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, best_mAP_50={best_mAP_50:.4f}, best_mAP_50_95={best_mAP_50_95:.4f}")

    print(f"\n{'='*60}")
    print(f"  Starting Training")
    print(f"  Epochs: {start_epoch} -> {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.img_size}")
    print(f"  Base LR: {args.lr}")
    print(f"{'='*60}\n")

    # Training log
    log_path = os.path.join(args.checkpoint_dir, 'training_log.json')
    training_log = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        stage = get_augment_stage(epoch, args.epochs)
        train_augment.set_stage(stage)

        raw_model = get_raw_model(model)
        if epoch < args.warmup_epochs:
            raw_model.freeze_backbone_stem()
        elif epoch == args.warmup_epochs:
            raw_model.unfreeze_backbone()
            print(f"  [Epoch {epoch}] Backbone unfrozen!")

        current_lrs = [g['lr'] for g in optimizer.param_groups]
        max_lr = max(current_lrs)

        print(f"Epoch {epoch}/{args.epochs-1} | Stage: {stage} | "
              f"LR: {max_lr:.2e}")

        try:
            train_losses = train_one_epoch(
                model, train_loader, criterion, optimizer, device,
                epoch, args.grad_clip
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  [ERROR] CUDA OOM at epoch {epoch}! Saving checkpoint and continuing...")
                torch.cuda.empty_cache()
                ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
                save_checkpoint(model, optimizer, scheduler, epoch,
                                best_val_loss, best_mAP_50, best_mAP_50_95, ckpt_path, is_best=False)
                continue
            else:
                raise

        scheduler.step()

        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            val_losses = validate(model, val_loader, criterion, device)
            val_loss = val_losses.get('total_loss', float('inf'))

            # Compute mAP on validation set
            print("  Evaluating mAP...")
            map_metrics = evaluate_map(model, val_loader, device, num_classes=5)
            mAP_50 = map_metrics['mAP_50']
            mAP_50_95 = map_metrics['mAP_50_95']

            print(f"  Val: loss={val_loss:.4f} "
                  f"(cls={val_losses.get('loss_cls', 0):.4f} "
                  f"reg={val_losses.get('loss_reg', 0):.4f} "
                  f"ctr={val_losses.get('loss_ctr', 0):.4f}) "
                  f"N_pos={val_losses.get('N_pos', 0):.0f}")
            print(f"  Val mAP: mAP@0.5={mAP_50:.4f} mAP@0.5:0.95={mAP_50_95:.4f}")

            # Check for best model
            if args.early_stop_metric == 'val_loss':
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    print(f"  ** New best val loss: {val_loss:.4f} **")
                else:
                    epochs_no_improve += 1
            elif args.early_stop_metric == 'mAP_50':
                is_best = mAP_50 > best_mAP_50
                if is_best:
                    best_mAP_50 = mAP_50
                    epochs_no_improve = 0
                    print(f"  ** New best mAP@0.5: {mAP_50:.4f} **")
                else:
                    epochs_no_improve += 1
            elif args.early_stop_metric == 'mAP_50_95':
                is_best = mAP_50_95 > best_mAP_50_95
                if is_best:
                    best_mAP_50_95 = mAP_50_95
                    epochs_no_improve = 0
                    print(f"  ** New best mAP@0.5:0.95: {mAP_50_95:.4f} **")
                else:
                    epochs_no_improve += 1

            # Keep other metric bests updated
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if mAP_50 > best_mAP_50:
                best_mAP_50 = mAP_50
            if mAP_50_95 > best_mAP_50_95:
                best_mAP_50_95 = mAP_50_95

            # Save checkpoint
            ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_loss, best_mAP_50, best_mAP_50_95,
                            ckpt_path, is_best=is_best)

            # Early stopping check
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\n  Early stopping triggered! "
                      f"No improvement in {args.early_stop_metric} for {args.patience} validation checks.")
                break
        else:
            val_losses = None
            map_metrics = None
            ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
            save_checkpoint(model, optimizer, scheduler, epoch,
                            best_val_loss, best_mAP_50, best_mAP_50_95,
                            ckpt_path, is_best=False)

        elapsed = time.time() - epoch_start
        log_entry = {
            'epoch': epoch,
            'stage': stage,
            'lr': max_lr,
            'train_losses': train_losses,
            'elapsed_sec': elapsed,
        }
        if val_losses is not None:
            log_entry['val_losses'] = val_losses
            log_entry['best_val_loss'] = best_val_loss
            if map_metrics is not None:
                log_entry['val_mAP_50'] = map_metrics['mAP_50']
                log_entry['val_mAP_50_95'] = map_metrics['mAP_50_95']
                log_entry['best_mAP_50'] = best_mAP_50
                log_entry['best_mAP_50_95'] = best_mAP_50_95

        training_log.append(log_entry)

        print(f"  Train: loss={train_losses.get('total_loss', 0):.4f} "
              f"(cls={train_losses.get('loss_cls', 0):.4f} "
              f"reg={train_losses.get('loss_reg', 0):.4f} "
              f"ctr={train_losses.get('loss_ctr', 0):.4f}) "
              f"IoU={train_losses.get('mean_iou', 0):.4f} "
              f"N_pos={train_losses.get('N_pos', 0):.0f} "
              f"| {elapsed:.1f}s")
        sys.stdout.flush()  

        # Save log periodically
        if (epoch + 1) % 5 == 0:
            with open(log_path, 'w') as f:
                json.dump(training_log, f, indent=2)

    ckpt_path = os.path.join(args.checkpoint_dir, 'last.pth')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1,
                    best_val_loss, best_mAP_50, best_mAP_50_95,
                    ckpt_path, is_best=False)

    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Best val loss: {best_val_loss:.4f} | Best mAP@0.5: {best_mAP_50:.4f} | Best mAP@0.5:0.95: {best_mAP_50_95:.4f}")
    print(f"  Best model saved to: {os.path.join(args.checkpoint_dir, 'best.pth')}")
    print(f"  Training log saved to: {log_path}")
    print(f"{'='*60}")
if __name__ == '__main__':
    main()