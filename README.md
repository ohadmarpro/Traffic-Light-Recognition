# 🚦 Traffic Light Detection & Classification – YOLO vs YOLO+CNN

This repository presents a smart, modular system for detecting traffic lights and classifying their signal state (**red**, **yellow**, **green**). It compares two approaches:

1. **YOLO-only**: Detection and classification combined within the YOLO bounding box model.
2. **YOLO + CNN**: YOLO handles detection only, and a separate CNN performs classification.

The project demonstrates the advantages of separating the detection and decision-making stages for better robustness, accuracy, and flexibility.

---

## 🧠 Project Motivation

In real-world scenarios like autonomous driving or traffic monitoring, traffic lights appear as **small**, **distant**, and sometimes **partially occluded** objects. Many models struggle to detect them reliably — and even when detected, classifying the **light signal** can be tricky under shadows, night conditions, or glare.

To address this, I built a **hybrid system** that mimics human behavior:

> First we locate the traffic light, then we look carefully to identify the active color.

---

## 🧩 Tools & Technologies Used

| Component       | Tool/Framework       | Purpose                                                             |
| --------------- | -------------------- | ------------------------------------------------------------------- |
| Detection       | YOLOv8 (Ultralytics) | Fast and accurate traffic light detection in images and video       |
| Classification  | PyTorch + Custom CNN | Lightweight classifier to determine signal state (red/yellow/green) |
| Data Processing | OpenCV, NumPy        | Image cropping, preprocessing, resizing                             |
| Visualization   | Matplotlib, Pillow   | Annotated output images and GIF generation                          |
| Automation      | Python scripts       | Batch processing and evaluation pipeline                            |

---

## 🧪 Methodology Overview

### 1. **YOLO-only Pipeline**

* A single YOLOv8 model is trained to detect traffic lights **and** predict their state directly.
* Works well when lights are clearly visible.
* Struggles with complex backgrounds, bright sunlight, or occlusions.

### 2. **YOLO + CNN Hybrid Pipeline**

* YOLOv8 detects bounding boxes around traffic lights only.
* Each cropped light is passed to a separate CNN trained on labeled crops.
* Classification is more accurate and consistent.
* Allows tuning detection and classification models independently.

---

## 📊 Quantitative Comparison (Confidence = 0.8)

| Metric                  | YOLO-only                            | YOLO + CNN                     |
| ----------------------- | ------------------------------------ | ------------------------------ |
| Detection Accuracy      | Medium (misses some small lights)    | High (better generalization)   |
| Classification Accuracy | Lower (misclassifies similar colors) | High (color-specific training) |
| False Positives         | Moderate                             | Very low                       |
| Flexibility             | Low                                  | High (modular)                 |
| Interpretability        | Hard to debug                        | Easy to separate issues        |

---

## 🖼️ Visual Results (conf threshold = 0.8)

### 🔹 YOLO-only Example:

![YOLO_Only – conf=0.8](Traffic-Light-Recognition/yolo_only/result_conf_08.png)

### 🔸 YOLO + CNN Example:

![YOLO&CNN – conf=0.8](Traffic-Light-Recognition/yolo&cnn/result_conf_08.png)

> The CNN-enhanced pipeline clearly performs better in edge cases, especially with occluded or dim signals.

---

## ▶ How to Run

```bash
pip install -r requirements.txt
python src/detect_and_classify.py --source your_video.mp4
```

* The script will run both YOLO and CNN stages.
* Annotated video or frames will be saved in `outputs/`

---

## 📂 Project Structure

```
traffic-light-detector/
├── src/
│   ├── detect_and_classify.py
│   ├── yolo_pipeline.py
│   └── classifier.py
├── comparisons/
│   ├── yolo_only/result_conf_08.png
│   └── yolo_plus_cnn/result_conf_08.png
├── models/
│   └── best.pt  # YOLO weights
├── runs/
│   └── classified_outputs/  # annotated images
├── requirements.txt
└── README.md
```

---

## 🙋‍♂️ About Me

Created by **Ohad Marhozi**, Electrical Engineering student with a focus on embedded systems, computer vision and deep learning. I built this system to explore modularity in vision pipelines and demonstrate how simple tools, when separated wisely, can outperform complex end-to-end models.

For collaborations or questions, feel free to reach out!

---

📅 2025 · Smart Mobility · Deep Learning · Vision Systems
