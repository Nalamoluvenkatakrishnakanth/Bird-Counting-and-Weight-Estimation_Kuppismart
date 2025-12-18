from fastapi import FastAPI, UploadFile, File
import cv2
import os
import uuid
import random
from ultralytics import YOLO

app = FastAPI()

# Load YOLOv8 model (COCO – 80 classes)
model = YOLO("yolov8n.pt")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# AUTO-GENERATE COLORS FOR ALL CLASSES
# ==============================
def generate_class_colors(class_names):
    random.seed(42)
    colors = {}
    for name in class_names.values():
        colors[name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return colors

CLASS_COLORS = generate_class_colors(model.names)

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
async def analyze_video(video: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    input_path = os.path.join(OUTPUT_DIR, f"{video_id}_{video.filename}")

    # Save uploaded video
    with open(input_path, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(OUTPUT_DIR, f"annotated_{video_id}.mp4")

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    # ==============================
    # TRACKING + COUNTING
    # ==============================
    seen_track_ids = set()
    class_counts = {}
    bird_areas = {}  # track_id -> list of bounding box areas

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.4)[0]

        if results.boxes.id is not None:
            for box, track_id in zip(results.boxes, results.boxes.id):
                track_id = int(track_id)
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(class_name, (255, 255, 255))

                # ==============================
                # COUNT ONLY ONCE
                # ==============================
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                # ==============================
                # COLLECT BIRD AREA
                # ==============================
                area = (x2 - x1) * (y2 - y1)
                if class_name == "bird":
                    if track_id not in bird_areas:
                        bird_areas[track_id] = []
                    bird_areas[track_id].append(area)

                # ==============================
                # CALCULATE WEIGHT INDEX FOR DISPLAY
                # ==============================
                if bird_areas:
                    global_avg_area = sum([sum(a)/len(a) for a in bird_areas.values()]) / len(bird_areas)
                else:
                    global_avg_area = 1.0

                weight_index = round(area / global_avg_area, 2)

                # ==============================
                # DRAW BOUNDING BOX + LABEL + ID + WEIGHT
                # ==============================
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{class_name} ID:{track_id} W:{weight_index}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

        # ==============================
        # VERTICAL COUNT OVERLAY
        # ==============================
        start_y = 40
        gap = 28
        for idx, (cls, cnt) in enumerate(class_counts.items(), start=1):
            y = start_y + (idx - 1) * gap
            cv2.rectangle(frame, (5, y - 22), (300, y + 6), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"{idx}. {cls} : {cnt}",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        out.write(frame)

    cap.release()
    out.release()

    # ==============================
    # FINAL WEIGHT PROXY CALCULATION
    # ==============================
    avg_area_per_bird = {tid: sum(areas)/len(areas) for tid, areas in bird_areas.items()}
    if avg_area_per_bird:
        global_avg_area = sum(avg_area_per_bird.values()) / len(avg_area_per_bird)
    else:
        global_avg_area = 1.0

    weight_index = {str(tid): round(area/global_avg_area, 2) for tid, area in avg_area_per_bird.items()}

    # ==============================
    # RETURN JSON
    # ==============================
    return {
        "message": "Video processed successfully",
        "counts": {
            "unique_object_counts": class_counts,
            "total_unique_objects": len(seen_track_ids)
        },
        "tracks_sample": [
            {"track_id": tid, "weight_index": weight_index.get(tid)}
            for tid in list(weight_index.keys())[:5]
        ],
        "weight_estimates": {
            "unit": "relative_weight_index",
            "per_bird": weight_index,
            "note": (
                "This is a relative weight proxy based on bounding box area. "
                "To convert to grams, camera calibration (pixel-to-cm), "
                "fixed camera height, and at least one known bird weight "
                "are required."
            )
        },
        "artifacts": {
            "input_video": input_path,
            "annotated_video": output_path
        }
    }
