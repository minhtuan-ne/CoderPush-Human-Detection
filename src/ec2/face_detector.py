import cv2
from datetime import datetime, timezone
import os
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    def __init__(self, output_dir="detected_faces", tolerance=0.6):
        self.face_counter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tolerance = tolerance
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def process_frame(self, frame):
        faces = self.face_app.get(frame)
        results = []
        for face in faces:
            embedding = face.embedding
            # Deduplication logic removed: save every detected face
            self.face_counter += 1
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            filepath = self._crop_and_save_face(frame, y1, x2, y2, x1, self.face_counter, timestamp)
            if not filepath:
                continue
            results.append({
                "face_id": self.face_counter,
                "img_URL": filepath,
                "timestamp": timestamp
            })
        print(results)
        return results

    def process_video_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame or reached end of video")
                break
            self.process_frame(frame)
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def _crop_and_save_face(self, frame, top, right, bottom, left, face_id, timestamp):
        margin = 30
        img_height, img_width = frame.shape[:2]
        crop_top = max(0, top - margin)
        crop_right = min(img_width, right + margin)
        crop_bottom = min(img_height, bottom + margin)
        crop_left = max(0, left - margin)
        face_img = frame[crop_top:crop_bottom, crop_left:crop_right]
        safe_timestamp = timestamp.replace(":", "-")
        filename = f"face_{face_id}_{safe_timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        success = cv2.imwrite(filepath, face_img)
        return filepath if success else None

# if __name__ == "__main__":
#     detector = FaceDetector()
#     frame = cv2.imread('detected_faces/sample_frame.png')
#     if frame is None:
#         print("Failed to load image!")
#     else:
#         detector.process_frame(frame)
