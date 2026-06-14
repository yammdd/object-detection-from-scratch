# Hướng dẫn chạy mô hình FCOS Object Detection

Tài liệu hướng dẫn cài đặt môi trường, huấn luyện, và chạy dự đoán cho mô hình phát hiện đối tượng FCOS từ đầu (from scratch).

---

## 📂 Cấu trúc mã nguồn quan trọng

*   `object-detection-from-scratch/predict.py`: File chạy dự đoán (inference).
*   `object-detection-from-scratch/train.py`: File chạy huấn luyện (training).
*   `object-detection-from-scratch/models/`: Chứa mã nguồn mô hình và tệp trọng số `best.pth`.
*   `object-detection-from-scratch/requirements.txt`: Danh sách thư viện Python cần cài đặt.

---

## 1. Cách cài đặt môi trường

Chuẩn bị Python 3.10+ và chạy lệnh sau từ thư mục gốc dự án để cài đặt các thư viện phụ thuộc:

```bash
pip install -r object-detection-from-scratch/requirements.txt
```

---

## 2. Vị trí đặt trọng số mô hình

*   **Vị trí**: File trọng số huấn luyện sẵn tốt nhất phải được đặt tại thư mục:
    `object-detection-from-scratch/models/best.pth`
*   **Lưu ý**: Nếu không tìm thấy file cục bộ tại đường dẫn trên, khi chạy `predict.py` mã nguồn sẽ **tự động** tải trọng số từ GitHub về thư mục này. Có thể tải thủ công tại: [Link tải best.pth](https://github.com/yammdd/object-detection-from-scratch/releases/download/FCOS/best.pth).

---

## 3. Cách chạy suy luận (Predict / Inference)

Trước khi chạy, hãy di chuyển vào thư mục `object-detection-from-scratch`:

```bash
cd object-detection-from-scratch
```

Chạy lệnh suy luận trên thư mục hình ảnh:

```bash
python predict.py --image_dir <đường_dẫn_thư_mục_ảnh> --output predictions.json
```

**Các tham số chính:**
*   `--image_dir` (Bắt buộc): Đường dẫn đến thư mục chứa ảnh cần dự đoán.
*   `--output` (Mặc định: `predictions.json`): File JSON đầu ra lưu kết quả dự đoán.
*   `--conf_threshold` (Mặc định: `0.3`): Ngưỡng độ tin cậy của đối tượng.
*   `--iou_threshold` (Mặc định: `0.5`): Ngưỡng IoU sử dụng trong NMS.
*   `--batch_size` (Mặc định: `32`): Batch size khi xử lý ảnh.

---

## 4. Cách huấn luyện (Training)

Di chuyển vào thư mục `object-detection-from-scratch`:

```bash
cd object-detection-from-scratch
```

Chạy lệnh huấn luyện với các đường dẫn dữ liệu tương ứng:

```bash
python train.py \
  --train_data public/annotations/train.json \
  --val_data public/annotations/val.json \
  --image_dir public/train/images \
  --val_image_dir public/val/images \
  --checkpoint_dir models/ \
  --epochs 100 \
  --batch_size 16 \
  --lr 1e-3
```