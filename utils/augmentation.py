import numpy as np
import cv2
import random
import math


def clip_bboxes(bboxes, img_w, img_h, min_area=20, min_side=2):
    bboxes = bboxes.copy()
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, img_w)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, img_h)

    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]
    valid = (w >= min_side) & (h >= min_side) & (w * h >= min_area)

    return valid, bboxes


def compute_iou_single(box, boxes):
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(x1 - x0, 0) * np.maximum(y1 - y0, 0)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter + 1e-7

    return inter / union


def random_horizontal_flip(img, bboxes, class_ids, p=0.5):
    if random.random() < p:
        h, w = img.shape[:2]
        img = img[:, ::-1, :].copy()  # horizontal flip

        if len(bboxes) > 0:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 0] = w - x1
            bboxes[:, 2] = w - x0

    return img, bboxes, class_ids


def random_hsv(img, bboxes, class_ids,
               h_gain=18, s_gain=0.5, v_gain=0.5):
    h_shift = random.uniform(-h_gain, h_gain)
    s_scale = random.uniform(1 - s_gain, 1 + s_gain)
    v_scale = random.uniform(1 - v_gain, 1 + v_gain)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img, bboxes, class_ids


def random_scale_crop(img, bboxes, class_ids, p=0.5,
                      scale_range=(0.5, 1.5), min_iou=0.3, max_tries=50):
    if random.random() >= p or len(bboxes) == 0:
        return img, bboxes, class_ids

    h, w = img.shape[:2]
    scale = random.uniform(*scale_range)

    new_w = int(w * scale)
    new_h = int(h * scale)
    new_w = max(new_w, 32)
    new_h = max(new_h, 32)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Scale bboxes
    scaled_bboxes = bboxes.copy()
    scaled_bboxes[:, [0, 2]] *= (new_w / w)
    scaled_bboxes[:, [1, 3]] *= (new_h / h)

    # Crop dimensions = original size
    crop_w = min(w, new_w)
    crop_h = min(h, new_h)

    # Try to find a valid crop
    for _ in range(max_tries):
        x0 = random.randint(0, max(0, new_w - crop_w))
        y0 = random.randint(0, max(0, new_h - crop_h))
        crop_box = np.array([x0, y0, x0 + crop_w, y0 + crop_h], dtype=np.float32)

        # Check overlap with bboxes
        ious = compute_iou_single(crop_box, scaled_bboxes)
        if ious.max() >= min_iou:
            # Crop image
            cropped = resized[y0:y0 + crop_h, x0:x0 + crop_w].copy()

            # Adjust bboxes
            new_bboxes = scaled_bboxes.copy()
            new_bboxes[:, [0, 2]] -= x0
            new_bboxes[:, [1, 3]] -= y0

            # Clip and filter
            valid, new_bboxes = clip_bboxes(new_bboxes, crop_w, crop_h)
            new_bboxes = new_bboxes[valid]
            new_class_ids = class_ids[valid]

            if len(new_bboxes) > 0:
                # Resize crop back to original size if needed
                if cropped.shape[:2] != (h, w):
                    sx = w / crop_w
                    sy = h / crop_h
                    cropped = cv2.resize(cropped, (w, h),
                                         interpolation=cv2.INTER_LINEAR)
                    new_bboxes[:, [0, 2]] *= sx
                    new_bboxes[:, [1, 3]] *= sy
                return cropped, new_bboxes, new_class_ids

    # If no valid crop found, return original
    return img, bboxes, class_ids


def random_cutout(img, bboxes, class_ids, p=0.3,
                  num_patches=(1, 3), max_ratio=0.2):
    if random.random() >= p:
        return img, bboxes, class_ids

    h, w = img.shape[:2]
    img = img.copy()

    n = random.randint(*num_patches)
    for _ in range(n):
        # Patch size
        pw = random.randint(int(w * 0.05), int(w * math.sqrt(max_ratio)))
        ph = random.randint(int(h * 0.05), int(h * math.sqrt(max_ratio)))
        # Patch position
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        x0 = max(0, cx - pw // 2)
        y0 = max(0, cy - ph // 2)
        x1 = min(w, cx + pw // 2)
        y1 = min(h, cy + ph // 2)

        # Fill with gray (114, 114, 114)
        img[y0:y1, x0:x1] = 114

    return img, bboxes, class_ids


def random_grayscale(img, bboxes, class_ids, p=0.05):
    if random.random() < p:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return img, bboxes, class_ids


class MosaicAugmentation:
    def __init__(self, dataset, img_size=416, p=0.5):
        self.dataset = dataset
        self.img_size = img_size
        self.p = p

    def __call__(self, img, bboxes, class_ids, idx=None):
        if random.random() >= self.p:
            return img, bboxes, class_ids

        s = self.img_size

        # Random center point for the mosaic junction
        cx = int(random.uniform(s * 0.25, s * 0.75))
        cy = int(random.uniform(s * 0.25, s * 0.75))

        # Canvas
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)

        # Current image is index 0; sample 3 others
        n = len(self.dataset)
        indices = [idx if idx is not None else random.randint(0, n - 1)]
        indices += [random.randint(0, n - 1) for _ in range(3)]

        all_bboxes = []
        all_class_ids = []

        for i, data_idx in enumerate(indices):
            # Load raw image and annotations
            raw_img, raw_bboxes, raw_cls, _ = self.dataset.get_raw_image(data_idx)
            ih, iw = raw_img.shape[:2]

            # Define placement regions for each quadrant
            if i == 0:    # top-left
                # Destination on canvas
                dx0, dy0 = max(0, cx - iw), max(0, cy - ih)
                dx1, dy1 = cx, cy
            elif i == 1:  # top-right
                dx0, dy0 = cx, max(0, cy - ih)
                dx1, dy1 = min(s, cx + iw), cy
            elif i == 2:  # bottom-left
                dx0, dy0 = max(0, cx - iw), cy
                dx1, dy1 = cx, min(s, cy + ih)
            else:         # bottom-right
                dx0, dy0 = cx, cy
                dx1, dy1 = min(s, cx + iw), min(s, cy + ih)

            # Region dimensions on canvas
            rw = dx1 - dx0
            rh = dy1 - dy0
            if rw <= 0 or rh <= 0:
                continue

            # Resize raw image to fit the region
            scale_x = rw / iw
            scale_y = rh / ih
            scale = min(scale_x, scale_y)
            nw = int(iw * scale)
            nh = int(ih * scale)
            if nw <= 0 or nh <= 0:
                continue

            resized_img = cv2.resize(raw_img, (nw, nh),
                                     interpolation=cv2.INTER_LINEAR)

            # Paste onto canvas (center within the region)
            paste_x = dx0 + (rw - nw) // 2
            paste_y = dy0 + (rh - nh) // 2
            canvas[paste_y:paste_y + nh, paste_x:paste_x + nw] = resized_img

            # Transform bboxes
            if len(raw_bboxes) > 0:
                transformed = raw_bboxes.copy()
                transformed[:, [0, 2]] = transformed[:, [0, 2]] * scale + paste_x
                transformed[:, [1, 3]] = transformed[:, [1, 3]] * scale + paste_y
                all_bboxes.append(transformed)
                all_class_ids.append(raw_cls)

        # Merge all bboxes
        if len(all_bboxes) > 0:
            merged_bboxes = np.concatenate(all_bboxes, axis=0)
            merged_cls = np.concatenate(all_class_ids, axis=0)

            # Clip and filter
            valid, merged_bboxes = clip_bboxes(merged_bboxes, s, s)
            merged_bboxes = merged_bboxes[valid]
            merged_cls = merged_cls[valid]
        else:
            merged_bboxes = np.zeros((0, 4), dtype=np.float32)
            merged_cls = np.zeros((0,), dtype=np.int64)

        return canvas, merged_bboxes, merged_cls


class MixUpAugmentation:
    def __init__(self, dataset, p=0.15):
        self.dataset = dataset
        self.p = p

    def __call__(self, img, bboxes, class_ids, idx=None):
        if random.random() >= self.p:
            return img, bboxes, class_ids

        # Sample a random partner
        n = len(self.dataset)
        partner_idx = random.randint(0, n - 1)
        partner_img, partner_bboxes, partner_cls, _ = \
            self.dataset.get_raw_image(partner_idx)

        # Resize partner to match current image size
        h, w = img.shape[:2]
        partner_img = cv2.resize(partner_img, (w, h),
                                  interpolation=cv2.INTER_LINEAR)

        # Scale partner bboxes
        ph, pw = partner_img.shape[:2]
        if len(partner_bboxes) > 0:
            # Partner bboxes are in original coords; scale to (w, h)
            orig_ph, orig_pw = self.dataset.entries[partner_idx]['orig_h'], \
                               self.dataset.entries[partner_idx]['orig_w']
            partner_bboxes[:, [0, 2]] *= (w / orig_pw)
            partner_bboxes[:, [1, 3]] *= (h / orig_ph)

        # Alpha from Beta distribution (concentrated around 0.5)
        alpha = np.random.beta(32, 32)

        # Blend images
        mixed = (alpha * img.astype(np.float32) +
                 (1 - alpha) * partner_img.astype(np.float32))
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)

        # Concatenate targets (not blending labels!)
        if len(partner_bboxes) > 0:
            all_bboxes = np.concatenate([bboxes, partner_bboxes], axis=0) \
                if len(bboxes) > 0 else partner_bboxes
            all_cls = np.concatenate([class_ids, partner_cls], axis=0) \
                if len(class_ids) > 0 else partner_cls
        else:
            all_bboxes = bboxes
            all_cls = class_ids

        return mixed, all_bboxes, all_cls


class TrainAugmentation:
    def __init__(self, dataset=None, img_size=416, stage='warmup'):
        self.stage = stage
        self.img_size = img_size
        self.mosaic = None
        self.mixup = None
        if dataset is not None:
            self.mosaic = MosaicAugmentation(dataset, img_size=img_size, p=0.5)
            self.mixup = MixUpAugmentation(dataset, p=0.15)

    def set_stage(self, stage):
        self.stage = stage
        if self.mosaic is not None:
            if stage == 'finetune':
                self.mosaic.p = 0.0
            elif stage == 'warmup':
                self.mosaic.p = 0.0
            else:
                self.mosaic.p = 0.5

    def __call__(self, img, bboxes, class_ids, idx=None):
        # Mosaic (stage: main, advanced)
        if self.stage in ('main', 'advanced') and self.mosaic is not None:
            img, bboxes, class_ids = self.mosaic(img, bboxes, class_ids, idx=idx)

        # Spatial transforms
        img, bboxes, class_ids = random_horizontal_flip(img, bboxes, class_ids, p=0.5)

        if self.stage in ('main', 'advanced'):
            img, bboxes, class_ids = random_scale_crop(
                img, bboxes, class_ids, p=0.3)

        # Color transforms
        img, bboxes, class_ids = random_hsv(img, bboxes, class_ids)

        if self.stage != 'finetune':
            img, bboxes, class_ids = random_grayscale(
                img, bboxes, class_ids, p=0.05)

        # Advanced augmentations
        if self.stage == 'advanced':
            img, bboxes, class_ids = random_cutout(
                img, bboxes, class_ids, p=0.3)

            if self.mixup is not None:
                img, bboxes, class_ids = self.mixup(
                    img, bboxes, class_ids, idx=idx)

        return img, bboxes, class_ids