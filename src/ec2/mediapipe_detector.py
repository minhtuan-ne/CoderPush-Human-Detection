import cv2
import mediapipe as mp
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.face_embedding_util import FaceNetEmbedding

# Mediapipe setup
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use full range model: model_selection=1
face_detector = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

# Embedding model - create instance once
embedding_model = FaceNetEmbedding()

def mediapipe_detector(frame):
    """
    Detect faces using MediaPipe (full range) and extract embeddings.
    Returns list of face objects with bbox and embedding.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb_frame)

    faces = []
    if not results.detections:
        print("[INFO] No faces detected.")
        return faces

    img_height, img_width = frame.shape[:2]

    for i, detection in enumerate(results.detections):
        bboxC = detection.location_data.relative_bounding_box
        x1 = max(0, min(int(bboxC.xmin * img_width), img_width))
        y1 = max(0, min(int(bboxC.ymin * img_height), img_height))
        x2 = max(0, min(int((bboxC.xmin + bboxC.width) * img_width), img_width))
        y2 = max(0, min(int((bboxC.ymin + bboxC.height) * img_height), img_height))

        if x2 <= x1 or y2 <= y1:
            print(f"[WARN] Invalid bbox for face {i}: [{x1},{y1},{x2},{y2}]")
            continue

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"[WARN] Empty face crop for face {i}")
            continue

        embedding = embedding_model.get_embedding(face_crop)
        if embedding is None:
            print(f"[WARN] Failed to extract embedding for face {i}")
            continue

        face = type("Face", (), {})()
        face.bbox = np.array([x1, y1, x2, y2])
        face.embedding = embedding
        face.confidence = detection.score[0] if detection.score else 0.0
        face.crop = face_crop
        faces.append(face)

    return faces

def main():
    frame = cv2.imread("/workspaces/CoderPush-Human-Detection/saved_frames/frame_30.jpg")
    if frame is None:
        print("[ERROR] Failed to load test image.")
        return

    faces = mediapipe_detector(frame)
    print(f"[INFO] Final result: {len(faces)} face(s) detected.")

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Face {i+1}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(f"face_crop_{i+1}.jpg", face.crop)

    cv2.imwrite("detection_result.jpg", frame)
    print("[INFO] Saved detection_result.jpg and face crops.")

if __name__ == "__main__":
    main()
