# Poultry CCTV Video Analysis – FastAPI Prototype

## 📌 Overview

This project is a **prototype FastAPI service** that analyzes a fixed-camera CCTV video (e.g., poultry farm footage) to:

1. **Detect and count objects (especially birds)** over time using detection + tracking
2. **Avoid double-counting** by assigning stable tracking IDs
3. **Estimate bird weight using a proxy (Relative Weight Index)** when true weight ground truth is unavailable
4. **Generate an annotated output video** with bounding boxes, object names, and vertical count overlay

The implementation follows the problem statement requirements exactly and uses **YOLOv8 + built-in tracking (ByteTrack)**.

---

## Approach Summary

### 1️⃣ Detection

* Uses **YOLOv8 (COCO – 80 classes)** pretrained model
* Produces bounding boxes, class labels, and confidence scores

### 2️⃣ Tracking & Counting

* Uses **YOLOv8 tracking (ByteTrack)**
* Each object is assigned a **stable track ID**
* Objects are counted **only once per unique track ID**
* Prevents double-counting even if the object appears in multiple frames

### 3️⃣ Occlusion & ID Switch Handling

* ByteTrack handles short occlusions internally
* Counting is based on **first appearance of a new track ID**
* Minor ID switches may increase count slightly (noted limitation)

### 4️⃣ Weight Estimation (Proxy)

* Since true bird weights are not available, a **Relative Weight Index** is computed
* Proxy is based on **average bounding box area per tracked bird**

**Relative Weight Index formula:**

```
weight_index = bird_avg_area / global_avg_bird_area
```

This provides a **dimensionless weight proxy** useful for relative comparison.

👉 **To convert to grams**, the following are required:

* Pixel-to-centimeter calibration
* Fixed camera height and angle
* At least one bird with known real-world weight

---

## ⚙️ Setup Instructions

### 1️⃣ Create Virtual Environment

```bash
python -m venv fraud_env
fraud_env\Scripts\activate   # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install fastapi uvicorn opencv-python ultralytics
```

> ⚠️ Make sure you have **Python 3.9+** installed

---

## ▶️ Run the API

From the project root directory:

```bash
uvicorn poultry_ai.app:app --reload
```

Server will start at:

```
(http://127.0.0.1:8000/analyze_video)


## 🎬 Output Artifacts

* **Annotated video** includes:

  * Bounding boxes
  * Object names (red)
  * Consistent color per object class
  * Vertical object count overlay



## ⚠️ Limitations

* Weight estimation is **relative**, not absolute
* Severe occlusions may cause occasional ID switches
* Designed for **fixed-camera** scenarios

---

## ✅ Compliance Checklist

* ✔ Bird detection & counting
* ✔ Tracking with stable IDs
* ✔ Double-count prevention
* ✔ Weight proxy estimation (calibration-based pixel-to-real mapping).
* ✔ Annotated output video
* ✔ FastAPI service
* ✔ JSON response as specified

