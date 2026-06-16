import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["person", "car", "dog", "cat", "chair"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def letterbox(img, target_size, color=(114, 114, 114)):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    padded = np.full((target_size, target_size, 3), color, dtype=np.uint8)

    # Center the resized image on the canvas
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, scale, pad_w, pad_h


def transform_bboxes(bboxes, scale, pad_w, pad_h):
    if len(bboxes) == 0:
        return bboxes.copy()

    transformed = bboxes.copy().astype(np.float32)
    transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + pad_w
    transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + pad_h
    return transformed


def normalize_image(img):
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Normalize
    img_rgb = (img_rgb - IMAGENET_MEAN) / IMAGENET_STD
    # HWC -> CHW
    img_chw = np.transpose(img_rgb, (2, 0, 1))
    return torch.from_numpy(img_chw)


class DetectionDataset(Dataset):
    def __init__(self, ann_file, img_dir, img_size=416, augment=None):
        super().__init__()
        self.img_size = img_size
        self.augment = augment

        # Parse annotation JSON
        with open(ann_file, 'r') as f:
            data = json.load(f)

        self.classes = data['classes']

        # Build image info lookup
        self.images = []
        img_id_to_info = {}
        for img_info in data['images']:
            img_id = img_info['id']
            img_id_to_info[img_id] = img_info

        # Group annotations by image_id
        from collections import defaultdict
        ann_by_image = defaultdict(list)
        for ann in data['annotations']:
            ann_by_image[ann['image_id']].append(ann)

        # Build dataset entries
        self.entries = []
        for img_info in data['images']:
            img_id = img_info['id']
            file_name = img_info['file_name']
            path1 = os.path.join(img_dir, file_name)
            path2 = os.path.join(img_dir, os.path.basename(file_name))

            if os.path.exists(path1):
                img_path = path1
            elif os.path.exists(path2):
                img_path = path2
            else:
                continue

            # Parse annotations for this image
            anns = ann_by_image.get(img_id, [])
            bboxes = []
            class_ids = []
            for ann in anns:
                bbox = ann['bbox']  # [xmin, ymin, xmax, ymax]
                cls_name = ann['class']
                cls_id = CLASS_TO_ID.get(cls_name, -1)
                if cls_id < 0:
                    continue
                bboxes.append(bbox)
                class_ids.append(cls_id)

            entry = {
                'image_id': img_id,
                'img_path': img_path,
                'orig_w': img_info['width'],
                'orig_h': img_info['height'],
                'bboxes': np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                'class_ids': np.array(class_ids, dtype=np.int64),
            }
            self.entries.append(entry)

        print(f"[DetectionDataset] Loaded {len(self.entries)} images "
              f"from {ann_file}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load image (BGR)
        img = cv2.imread(entry['img_path'])
        if img is None:
            raise RuntimeError(f"Failed to load image: {entry['img_path']}")

        bboxes = entry['bboxes'].copy()       # (N, 4) [x0, y0, x1, y1]
        class_ids = entry['class_ids'].copy()  # (N,)

        # Apply augmentation
        if self.augment is not None:
            img, bboxes, class_ids = self.augment(img, bboxes, class_ids, idx=idx)

        # Letterbox resize
        img_lb, scale, pad_w, pad_h = letterbox(img, self.img_size)

        # Transform bboxes to letterbox coordinates
        bboxes_lb = transform_bboxes(bboxes, scale, pad_w, pad_h)

        # Clip bboxes to image boundaries
        bboxes_lb[:, [0, 2]] = np.clip(bboxes_lb[:, [0, 2]], 0, self.img_size)
        bboxes_lb[:, [1, 3]] = np.clip(bboxes_lb[:, [1, 3]], 0, self.img_size)

        # Filter out degenerate boxes
        if len(bboxes_lb) > 0:
            w = bboxes_lb[:, 2] - bboxes_lb[:, 0]
            h = bboxes_lb[:, 3] - bboxes_lb[:, 1]
            valid = (w > 2) & (h > 2)
            bboxes_lb = bboxes_lb[valid]
            class_ids = class_ids[valid]

        # Normalize image
        img_tensor = normalize_image(img_lb)

        # Build targets: (N, 5) = [class_id, x0, y0, x1, y1]
        if len(bboxes_lb) > 0:
            targets = np.concatenate([
                class_ids[:, None].astype(np.float32),
                bboxes_lb
            ], axis=1)
        else:
            targets = np.zeros((0, 5), dtype=np.float32)

        targets = torch.from_numpy(targets)

        meta = {
            'image_id': entry['image_id'],
            'orig_w': entry['orig_w'],
            'orig_h': entry['orig_h'],
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
        }

        return img_tensor, targets, meta

    def get_raw_image(self, idx):
        entry = self.entries[idx]
        img = cv2.imread(entry['img_path'])
        return img, entry['bboxes'].copy(), entry['class_ids'].copy(), entry['image_id']


def collate_fn(batch):
    images, targets, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets), list(metas)


class InferenceDataset(Dataset):
    def __init__(self, image_dir, img_size=416):
        self.img_size = img_size
        self.image_dir = image_dir
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ])
        print(f"[InferenceDataset] Found {len(self.image_files)} images "
              f"in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # Letterbox
        img_lb, scale, pad_w, pad_h = letterbox(img, self.img_size)

        # Normalize
        img_tensor = normalize_image(img_lb)

        meta = {
            'image_id': fname,
            'orig_w': orig_w,
            'orig_h': orig_h,
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
        }

        return img_tensor, meta


def inference_collate_fn(batch):
    images, metas = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(metas)