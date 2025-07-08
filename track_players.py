import cv2
import os
from ultralytics import YOLO

from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# ========== CONFIG ==========
MODEL_PATH = "best.pt"
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"
PLAYER_CLASS_ID = 2
CONFIDENCE_THRESHOLD = 0.3
# ============================

def load_model():
    return YOLO(MODEL_PATH)

def init_tracker():
    return DeepSort(max_age=30)

def draw_id(frame, bbox, track_id):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def process_video(video_path, model, tracker, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        results = model(frame)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == PLAYER_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'player'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            draw_id(frame, (l, t, r, b), track_id)

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Processed: {os.path.basename(video_path)}")

def main():
    model = load_model()
    tracker = init_tracker()
    video_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".mp4")]

    for file in video_files:
        input_path = os.path.join(INPUT_DIR, file)
        output_path = os.path.join(OUTPUT_DIR, f"tracked_{file}")
        process_video(input_path, model, tracker, output_path)

if __name__ == "__main__":
    main()
