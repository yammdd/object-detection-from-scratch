import os
import json
import argparse
import time
import urllib.request

DEFAULT_CHECKPOINT_URL = "https://github.com/yammdd/object-detection-from-scratch/releases/download/FCOS/best.pth"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.fcos_detector import FCOSDetector, FPN_STRIDES
from utils.dataset import (
    InferenceDataset, inference_collate_fn, CLASS_NAMES, NUM_CLASSES
)
from utils.decoder import decode_fcos_outputs, rescale_boxes
from utils.nms import post_process_single_image


def predict(model, dataloader, device, conf_threshold=0.3,
            iou_threshold=0.5, max_detections=300):
    model.eval()
    results = {}

    total_images = 0
    start_time = time.time()

    with torch.no_grad():
        for images, metas in dataloader:
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
                meta = metas[b]
                image_id = meta['image_id']

                # Post-process: threshold + NMS
                det_boxes, det_scores, det_classes = post_process_single_image(
                    all_boxes[b], all_scores[b], all_classes[b],
                    num_classes=NUM_CLASSES,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    max_detections=max_detections,
                )

                # Rescale boxes from letterboxed coords to original image coords
                if det_boxes.shape[0] > 0:
                    det_boxes, valid_mask = rescale_boxes(
                        det_boxes,
                        scale=meta['scale'],
                        pad_w=meta['pad_w'],
                        pad_h=meta['pad_h'],
                        orig_w=meta['orig_w'],
                        orig_h=meta['orig_h'],
                    )
                    det_boxes = det_boxes[valid_mask]
                    det_scores = det_scores[valid_mask]
                    det_classes = det_classes[valid_mask]

                # Format output
                boxes_out = []
                for i in range(det_boxes.shape[0]):
                    x0, y0, x1, y1 = det_boxes[i].tolist()
                    boxes_out.append({
                        "class": CLASS_NAMES[int(det_classes[i])],
                        "confidence": round(float(det_scores[i]), 4),
                        "bbox": [int(round(x0)), int(round(y0)),
                                 int(round(x1)), int(round(y1))]
                    })

                results[image_id] = boxes_out
                total_images += 1

    elapsed = time.time() - start_time
    fps = total_images / elapsed if elapsed > 0 else 0
    print(f"Inference complete: {total_images} images in {elapsed:.1f}s "
          f"({fps:.1f} FPS)")

    return results


def format_output(results, image_ids):
    output = []
    for image_id in image_ids:
        output.append({
            "image_id": image_id,
            "boxes": results.get(image_id, [])
        })
    return output


def download_checkpoint(url, dest_path):
    dest_dir = os.path.dirname(dest_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Local checkpoint not found. Downloading from {url}...")
    
    def progress_hook(count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = count * block_size
        percent = min(100.0, downloaded * 100.0 / total_size)
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # Display progress bar
        bar_length = 30
        filled_length = int(round(bar_length * percent / 100.0))
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r[{bar}] {percent:.1f}% ({downloaded_mb:.2f}MB / {total_mb:.2f}MB)", end='', flush=True)

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print("\nDownload complete successfully.")
    except Exception as e:
        print(f"\nError downloading checkpoint: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise e


def main():
    parser = argparse.ArgumentParser(
        description='FCOS Object Detection Inference')

    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing images for inference')
    parser.add_argument('--output', type=str, default='predictions.json',
                        help='Output predictions JSON path')
    parser.add_argument('--checkpoint', type=str, default='./models/best.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--checkpoint-url', type=str, default=DEFAULT_CHECKPOINT_URL,
                        help='URL to download the checkpoint if it does not exist locally')

    parser.add_argument('--img_size', type=int, default=512,
                        help='Input image size for letterbox')
    parser.add_argument('--conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for detections')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='IoU threshold for NMS')
    parser.add_argument('--max_detections', type=int, default=300,
                        help='Maximum detections per image')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    print(f"Device: {device}")
    if num_gpus > 0:
        for gpu_i in range(num_gpus):
            print(f"  GPU {gpu_i}: {torch.cuda.get_device_name(gpu_i)}")

    # Download checkpoint if it does not exist locally
    if not os.path.isfile(args.checkpoint):
        if args.checkpoint_url and args.checkpoint_url.startswith("http") and "your-username" not in args.checkpoint_url:
            try:
                download_checkpoint(args.checkpoint_url, args.checkpoint)
            except Exception as e:
                print(f"WARNING: Failed to download checkpoint from URL: {args.checkpoint_url}")
                print(f"Detail: {e}")
        else:
            print(f"Local checkpoint not found at '{args.checkpoint}' and no valid download URL is configured.")

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = FCOSDetector(num_classes=NUM_CLASSES, pretrained_backbone=False)

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device,
                                weights_only=False)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        clean_sd = {}
        for k, v in state_dict.items():
            clean_sd[k.replace('module.', '', 1) if k.startswith('module.') else k] = v
        model.load_state_dict(clean_sd)
    else:
        print(f"WARNING: Checkpoint not found at {args.checkpoint}")
        print("Running inference with random weights (for testing only)")

    model.to(device)

    # Multi-GPU inference
    if num_gpus > 1:
        print(f"Using DataParallel on {num_gpus} GPUs for inference")
        model = nn.DataParallel(model)

    model.eval()

    # Build inference dataset
    inf_dataset = InferenceDataset(args.image_dir, img_size=args.img_size)
    inf_loader = DataLoader(
        inf_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=inference_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
    )

    # Run inference
    print(f"\nRunning inference...")
    print(f"  Images: {len(inf_dataset)}")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")

    results = predict(
        model, inf_loader, device,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        max_detections=args.max_detections,
    )

    # Format output
    all_image_ids = inf_dataset.image_files
    output = format_output(results, all_image_ids)

    # Count statistics
    total_dets = sum(len(item['boxes']) for item in output)
    images_with_dets = sum(1 for item in output if len(item['boxes']) > 0)

    # Write output
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {os.path.abspath(args.output)}")
    print(f"  Total images: {len(output)}")
    print(f"  Images with detections: {images_with_dets}")
    print(f"  Total detections: {total_dets}")
    if len(output) > 0:
        print(f"  Avg detections/image: {total_dets / len(output):.1f}")


if __name__ == '__main__':
    main()