import cv2
from datetime import datetime, timezone
import os
import numpy as np
from insightface.app import FaceAnalysis
from zoneinfo import ZoneInfo
from numpy.linalg import norm
import time
from flask import jsonify

class FaceDetector:
    def __init__(self, output_dir="detected_faces", tolerance=0.6):
        self.face_counter = 0
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.tolerance = tolerance
        self.known_embeddings = []
        self.models = ['buffalo_l', 'buffalo_m', 'buffalo_s', 'buffalo_sc', 'antelopev2']
        self.current_model = self.models[0]
        try:
            self.face_app = FaceAnalysis(name=self.current_model, providers=['CPUExecutionProvider'], root="/tmp/.insightface")
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InsightFace: {str(e)}")

        self.frame_skip_counter = 0

    def get_local_timestamp(self, tz_str='Asia/Ho_Chi_Minh'):
        utc_now = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        local_time = utc_now.astimezone(ZoneInfo(tz_str))
        return local_time.strftime('%Y-%m-%dT%H:%M:%SZ')

    def is_duplicate(self, embedding):
        for known_emb in self.known_embeddings:
            sim = np.dot(embedding, known_emb) / (norm(embedding) * norm(known_emb))
            if sim > (1 - self.tolerance):
                print("Duplicate face")
                return True
        return False

    def process_frame(self, frame):
        self.frame_skip_counter += 1
        if self.frame_skip_counter % 20 != 0:
            print('skip')
            return []

        else:
            filename = f"frame_{self.frame_skip_counter}.jpg"
            filepath = os.path.join('saved_frames', filename)
            success = cv2.imwrite(filepath, frame)

        print("Processing frame", self.frame_skip_counter)

        faces = self.face_app.get(frame)
        results = []
        for face in faces:
            embedding = face.embedding
            if embedding is None:
                continue
            if self.is_duplicate(embedding):
                continue
            self.known_embeddings.append(embedding)
            self.face_counter += 1
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            timestamp = self.get_local_timestamp()
            filepath = self._crop_and_save_face(frame, y1, x2, y2, x1, self.face_counter, timestamp)
            if not filepath:
                continue
            results.append({
                "face_id": self.face_counter,
                "img_URL": filepath,
                "timestamp": timestamp
            })
        print(results)
        return jsonify(results)

    def process_video_stream(self, video_source=0, show_window=False, max_frames=10, emit_func=None):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")
        frame_count = 0
        all_results = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame_count >= max_frames:
                    break
                results = self.process_frame(frame)
                if emit_func:
                    emit_func(results) 
                    time.sleep(0.1) 

                all_results.append(results)

                print(all_results)

                frame_count += 1

                if show_window:
                    cv2.imshow('Face Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        except Exception as e:
            print(f"Error during video processing: {e}")
            raise
        finally:
            cap.release()
            if show_window:
                cv2.destroyAllWindows()

        return all_results


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

        # Compress image before saving
        success, encoded_img = cv2.imencode('.jpg', face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if success:
            with open(filepath, 'wb') as f:
                f.write(encoded_img)
            return filepath
        return None


# if __name__ == "__main__":
#     detector = FaceDetector()
#     # frame = cv2.imread('detected_faces/sample_frame.png')
#     # if frame is None:
#     #     print("Failed to load image!")
#     # else:
#         # detector.process_frame(frame)
#     video_source = 'live.ts'
#     detector.process_video_stream(video_source=video_source)